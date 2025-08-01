import tensorflow as tf

test_cases = [
    # Edge Case 1: Flip from long to short in one trade
    {
        "desc": "Long 1 → Sell 2 → Go short 1",
        "actions": [1.0, -2.0],
        "prices": [1.0, 0.9]
    },
    # Edge Case 2: Flip from short to long in one trade
    {
        "desc": "Short 1 → Buy 2 → Go long 1",
        "actions": [-1.0, 2.0],
        "prices": [1.0, 1.1]
    },
    # Edge Case 3: Buy and immediately sell at same price
    {
        "desc": "Buy 1 → Sell 1 at same price",
        "actions": [1.0, -1.0],
        "prices": [1.0, 1.0]
    },
    # Edge Case 4: Sell more than owned and re-enter same direction
    {
        "desc": "Long 1 → Sell 2 → Buy 1",
        "actions": [1.0, -2.0, 1.0],
        "prices": [1.0, 0.8, 1.1]
    },
    # Edge Case 5: Flat position, then action again
    {
        "desc": "Buy 1 → Sell 1 → Buy 1",
        "actions": [1.0, -1.0, 1.0],
        "prices": [1.0, 1.1, 1.2]
    },
    # Edge Case 6: All zero actions, but volatile prices
    {
        "desc": "No trades, price fluctuates",
        "actions": [0.0, 0.0, 0.0],
        "prices": [1.0, 2.0, 0.5]
    },
    # Edge Case 7: Fractional trades and rounding errors
    {
        "desc": "Tiny fractional trades",
        "actions": [0.01, -0.01],
        "prices": [100.0, 101.0]
    },
    # Edge Case 8: Round trip at a loss
    {
        "desc": "Buy 1 @ high → Sell 1 @ low",
        "actions": [1.0, -1.0],
        "prices": [1.2, 1.0]
    },
    # Edge Case 9: Reverse with neutral net position
    {
        "desc": "Buy 1 → Sell 2 → Buy 1 (net 0)",
        "actions": [1.0, -2.0, 1.0],
        "prices": [1.0, 0.9, 1.1]
    },
    # Edge Case 10: Many opens, single close
    {
        "desc": "Buy 0.5 + 0.5 + 1.0 → Sell 2.0",
        "actions": [0.5, 0.5, 1.0, -2.0],
        "prices": [1.0, 1.05, 1.1, 1.2]
    }
]

for i, case in enumerate(test_cases):
    actions = tf.convert_to_tensor(case["actions"], dtype=tf.float32)
    prices = tf.convert_to_tensor(case["prices"], dtype=tf.float32)
    
    pnl, unreal, txn, realized = calc_pnl(actions, prices)

    print(f"Case {i+1}: {case['desc']}")
    print(" Actions  :", actions.numpy())
    print(" Prices   :", prices.numpy())
    print(" PnL      :", pnl.numpy())
    print(" Realized :", realized.numpy())
    print(" Unreal.  :", unreal.numpy())
    print(" Txn Cost :", txn.numpy())
    print("-" * 40)
