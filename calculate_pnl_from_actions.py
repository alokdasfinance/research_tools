def compute_realized_pnl_stock(actions, prices):
    n = tf.shape(actions)[0]
    position_value = tf.constant(0.0, dtype=tf.float32)

    def body(i, position, cost_basis, realized_pnl, position_value):
        action = tf.cast(actions[i], tf.float32)
        price = tf.cast(prices[i], tf.float32)

        def trade():
            new_position = position + action

            def closing_trade():
                closing_size = tf.minimum(tf.abs(action), tf.abs(position))
                avg_price = tf.cond(
                    tf.not_equal(position, 0.0),
                    lambda: cost_basis / tf.abs(position),
                    lambda: tf.constant(0.0, dtype=tf.float32)
                )
                pnl = closing_size * (price - avg_price) * tf.sign(position)
                updated_position = position + action

                def reset_cost_basis():
                    return tf.abs(updated_position) * price

                updated_cost_basis = tf.cond(
                    tf.equal(tf.sign(position), tf.sign(updated_position)),
                    lambda: cost_basis - closing_size * avg_price,
                    reset_cost_basis
                )
                updated_position_value = tf.cond(
                    tf.equal(tf.sign(position), tf.sign(updated_position)),
                    lambda: position_value,
                    lambda: tf.abs(updated_position) * price
                )
                return updated_position, updated_cost_basis, realized_pnl + pnl, updated_position_value

            def opening_trade():
                new_cost_basis = cost_basis + tf.abs(action) * price
                updated_position_value = position_value + tf.abs(action) * price
                return new_position, new_cost_basis, realized_pnl, updated_position_value

            same_direction = tf.equal(tf.sign(position), tf.sign(action))
            return tf.cond(
                tf.logical_or(tf.equal(position, 0.0), same_direction),
                opening_trade,
                closing_trade
            )

        position, cost_basis, realized_pnl, position_value = trade()
        return i + 1, position, cost_basis, realized_pnl, position_value

    i0 = tf.constant(0)
    position0 = tf.constant(0.0, dtype=tf.float32)
    cost_basis0 = tf.constant(0.0, dtype=tf.float32)
    realized_pnl0 = tf.constant(0.0, dtype=tf.float32)
    position_value0 = tf.constant(0.0, dtype=tf.float32)

    _, final_position, final_cost_basis, final_realized_pnl, final_position_value = tf.while_loop(
        lambda i, *_: i < n,
        body,
        loop_vars=[i0, position0, cost_basis0, realized_pnl0, position_value0]
    )

    price_T = tf.cast(prices[-1], tf.float32)

    def safe_unrealized():
        avg_entry_price = tf.cond(
            tf.not_equal(final_position, 0.0),
            lambda: final_cost_basis / tf.abs(final_position),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )
        return tf.abs(final_position) * (price_T - avg_entry_price) * tf.sign(final_position)

    unrealized_pnl = tf.cond(
        tf.not_equal(final_position, 0.0),
        safe_unrealized,
        lambda: tf.constant(0.0, dtype=tf.float32)
    )

    return final_realized_pnl, unrealized_pnl
