def value_iteration(V_s, theta=0.01, discount_rate=0.5):
    value_for_state_map = create_value_for_state_map()  # 1.

    delta = 100  # 2.
    while not delta < theta:  # 3.
        delta = 0  # 4.
        for state in range(1, 15):  # 5.
            v = V_s[state]  # 6.

            totals = {}  # 7.
            for action in ["N", "S", "E", "W"]:
                total = 0
                for state_prime in range(16):
                    total += value_for_state_map[(state_prime, -1, state, action)] * (
                                -1 + discount_rate * V_s[state_prime])
                totals[action] = total

            V_s[state] = round(max(totals.values()), 4)  # 8.
            delta = max(delta, abs(v - V_s[state]))  # 9.
    return V_s  # 10.