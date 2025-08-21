import tensorflow as tf
from collections import defaultdict

test_cases = [
    {"desc": "Long 1 → Sell 2 → Go short 1",               "actions": [1.0, -2.0],             "prices": [1.0, 0.9]},
    {"desc": "Short 1 → Buy 2 → Go long 1",                "actions": [-1.0, 2.0],             "prices": [1.0, 1.1]},
    {"desc": "Buy 1 → Sell 1 at same price",               "actions": [1.0, -1.0],             "prices": [1.0, 1.0]},
    {"desc": "Long 1 → Sell 2 → Buy 1",                    "actions": [1.0, -2.0, 1.0],        "prices": [1.0, 0.8, 1.1]},
    {"desc": "Buy 1 → Sell 1 → Buy 1",                     "actions": [1.0, -1.0, 1.0],        "prices": [1.0, 1.1, 1.2]},
    {"desc": "No trades, price fluctuates",                "actions": [0.0, 0.0, 0.0],         "prices": [1.0, 2.0, 0.5]},
    {"desc": "Tiny fractional trades",                      "actions": [0.01, -0.01],           "prices": [100.0, 101.0]},
    {"desc": "Buy 1 @ high → Sell 1 @ low",                "actions": [1.0, -1.0],             "prices": [1.2, 1.0]},
    {"desc": "Buy 1 → Sell 2 → Buy 1 (net 0)",             "actions": [1.0, -2.0, 1.0],        "prices": [1.0, 0.9, 1.1]},
    {"desc": "Buy 0.5 + 0.5 + 1.0 → Sell 2.0",             "actions": [0.5, 0.5, 1.0, -2.0],   "prices": [1.0, 1.05, 1.1, 1.2]},
    {"desc": "unrealizedd",                                "actions": [1, 1, 0, 0],            "prices": [1.0, 1.05, 1.1, 1.2]}
]

single_results = []
for i, case in enumerate(test_cases):
    actions = tf.convert_to_tensor(case["actions"], dtype=tf.float32)  # (T,)
    prices  = tf.convert_to_tensor(case["prices"],  dtype=tf.float32)  # (T,)
    pnl, unreal, txn, realized = calc_pnl(actions, prices)
    single_results.append({
        "pnl": float(pnl.numpy()),
        "unreal": float(unreal.numpy()),
        "txn": float(txn.numpy()),
        "realized": float(realized.numpy()),
    })

groups = defaultdict(list)
for idx, case in enumerate(test_cases):
    groups[len(case["actions"])].append(idx)

batched_results = [None] * len(test_cases)
for T_len, idx_list in groups.items():
    # stack actions/prices for this group (B, T_len)
    actions_bt = tf.stack([tf.constant(test_cases[i]["actions"], tf.float32) for i in idx_list], axis=0)
    prices_bt  = tf.stack([tf.constant(test_cases[i]["prices"],  tf.float32) for i in idx_list], axis=0)

    pnl_b, unreal_b, txn_b, realized_b = calc_pnl_batched(actions_bt, prices_bt)  # (B,)

    # store per original index
    for k, idx in enumerate(idx_list):
        batched_results[idx] = {
            "pnl": float(pnl_b[k].numpy()),
            "unreal": float(unreal_b[k].numpy()),
            "txn": float(txn_b[k].numpy()),
            "realized": float(realized_b[k].numpy()),
        }

# 3) Print comparison for each case
tol = 1e-6
for i, case in enumerate(test_cases):
    s = single_results[i]
    b = batched_results[i]
    print(f"Case {i+1}: {case['desc']}")
    print(" Actions   :", case["actions"])
    print(" Prices    :", case["prices"])

    print(" Single    : PnL={:.8f}, Realized={:.8f}, Unreal={:.8f}, Txn={:.8f}".format(
        s["pnl"], s["realized"], s["unreal"], s["txn"]
    ))
    print(" Batched   : PnL={:.8f}, Realized={:.8f}, Unreal={:.8f}, Txn={:.8f}".format(
        b["pnl"], b["realized"], b["unreal"], b["txn"]
    ))

    ok = (abs(s["pnl"] - b["pnl"]) <= tol and
          abs(s["realized"] - b["realized"]) <= tol and
          abs(s["unreal"] - b["unreal"]) <= tol and
          abs(s["txn"] - b["txn"]) <= tol)
    print(" Match?    :", "PASS ✅" if ok else "DIFF ❌")
    if not ok:
        print("  Δ PnL     :", s["pnl"] - b["pnl"])
        print("  Δ Realized:", s["realized"] - b["realized"])
        print("  Δ Unreal  :", s["unreal"] - b["unreal"])
        print("  Δ Txn     :", s["txn"] - b["txn"])
    print("-" * 50)
