import numpy as np
import numpy.random as nr


def muscle_data_factory(num, tstop, dt, start_phase, phase_shift, half_period_t, amp, input_volt, amp_decay=None,
                        forward=True, seed=None, with_sign=False):
    if seed is not None:
        nr.seed(seed)
    len_sim, half_period = int(tstop / dt), int(half_period_t / dt)
    wave_traces, wave_signs = [], []
    shift_sign = 1 if forward else -1
    x = np.arange(len_sim)
    for i in range(num):
        tmp_wave = np.sin((x/half_period)*np.pi+start_phase-shift_sign*(i/num)*phase_shift)
        wave_signs.append(np.sign(tmp_wave))
        if amp_decay is not None:
            amp_decay_rate = amp_decay**(i/num) if forward else amp_decay**((num-i)/num)
        else:
            amp_decay_rate = 1
        wave_traces.append(amp*amp_decay_rate*(tmp_wave+1)/2)
    wave_traces = np.array(wave_traces)
    wave_signs = np.array(wave_signs)
    if with_sign:
        input_wave = (np.sin((x/half_period)*np.pi) + 1)/2 * input_volt[1] + input_volt[0]
        return wave_traces, wave_signs, input_wave
    else:
        return wave_traces


def direct(muscle_traces, directions, amps):
    assert len(directions) == muscle_traces.shape[-1] == len(amps)
    zeros, ones = np.zeros(24), np.ones(24)
    chill = {
        "L": np.concatenate([ones, ones, zeros, zeros]),
        "R": np.concatenate([zeros, zeros, ones, ones]),
        "D": np.concatenate([zeros, ones, zeros, ones]),
        "V": np.concatenate([ones, zeros, ones, zeros]),
    }
    for time_step, direction in enumerate(directions):
        if np.linalg.norm(direction) > 0.1:
            coef = direction / np.linalg.norm(direction)
            loose = (coef[0] * chill['L'] + coef[1] * chill['R'] + coef[2] * chill['D'] + coef[3] * chill['V']) * amps[
                time_step]
            muscle_traces[:, time_step] = muscle_traces[:, time_step] * (1 - loose)
    return muscle_traces


def dir_norm(direction):
    return direction / np.linalg.norm(direction)


def navigation(head_position, locomotion_direction, dorsal_direction, state_config=None, food_location=None):
    if food_location is None:
        food_location = np.zeros(3)
    locomotion_direction, dorsal_direction = dir_norm(locomotion_direction), dir_norm(dorsal_direction)
    target_position = dir_norm(food_location - head_position)
    left_direction = np.cross(dorsal_direction, locomotion_direction)
    direction = [0, 0, 0, 0]
    if np.dot(target_position, left_direction) > 0:
        direction[0] = np.dot(target_position, left_direction)
    else:
        direction[1] = -np.dot(target_position, left_direction)
    if np.dot(target_position, dorsal_direction) > 0:
        direction[2] = np.dot(target_position, dorsal_direction)
    else:
        direction[3] = -np.dot(target_position, dorsal_direction)
    amp = min(1 - np.dot(target_position, locomotion_direction), 0.4)

    if state_config is None:
        state_config = {
            "data_config": {
                "start_phase": -np.pi,
                "phase_shift": 5 * np.pi,
                "dt": 1,
                "tstop": 100,
                "half_period_t": 5,
                "amp": 0.8,
                "input_volt": (-30, 40),
                "amp_decay": 0.1,
                "forward": True}
        }

        state_config["data_config"]['start_phase'] = -np.pi
        test_muscle_13 = muscle_data_factory(num=24, **state_config["data_config"])

        state_config["data_config"]['start_phase'] = 0
        test_muscle_02 = muscle_data_factory(num=24, **state_config["data_config"])
        total_muscle = np.concatenate([test_muscle_02, test_muscle_13, test_muscle_02, test_muscle_13], axis=0)
        state_config["total_muscle"] = total_muscle
        state_config["phase"] = 0
    tmp_muscle = state_config["total_muscle"][:, state_config["phase"] % state_config["total_muscle"].shape[-1]].reshape(-1, 1)
    loosed_muscle = direct(tmp_muscle, [direction], [amp])
    state_config["phase"] += 1
    return loosed_muscle, state_config


def muscle_test_factory(sim_config, data_config):
    data_config['start_phase'] = -np.pi
    test_muscle_13 = eval("data_factory." + data_config['factory_name'])(
        num=24, tstop=sim_config['tstop'], dt=sim_config['dt'], **data_config)

    data_config['start_phase'] = 0
    test_muscle_02 = eval("data_factory." + data_config['factory_name'])(
        num=24, tstop=sim_config['tstop'], dt=sim_config['dt'], **data_config)

    quarter_len = int(test_muscle_02.shape[-1] / 5)
    directions = np.zeros((5 * quarter_len, 4))
    # directions[:quarter_len, 0] = 1
    directions[quarter_len:2 * quarter_len, 0] = 1
    directions[2 * quarter_len:3 * quarter_len, 1] = 1
    directions[3 * quarter_len:4 * quarter_len, 2] = 1
    directions[4 * quarter_len:5 * quarter_len, 3] = 1
    amps = np.ones(quarter_len * 5) * 0.2
    total_muscle = np.concatenate([test_muscle_02, test_muscle_13, test_muscle_02, test_muscle_13], axis=0)
    print(total_muscle.shape)
    total_muscle = direct(total_muscle, directions, amps)
    print(total_muscle.shape)
    return total_muscle


if __name__ == "__main__":
    pass
