"""Reject gross root-motion discontinuities before using a reference in physics."""
import numpy as np


def root_motion_report(values, fps=30.):
    x = np.asarray(values, dtype=float)
    if (x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 7
            or not np.isfinite(x[:, :7]).all() or not np.isfinite(fps) or fps <= 0):
        raise ValueError('Root motion requires at least two finite xyz/xyzw frames and positive fps')
    norms = np.linalg.norm(x[:, 3:7], axis=1, keepdims=True)
    if np.any(abs(norms - 1.) > 1e-5):
        raise ValueError('Root motion requires unit xyzw quaternions')
    speed = np.linalg.norm(np.diff(x[:, :3], axis=0), axis=1) * fps
    quat = x[:, 3:7] / norms
    angular = 2 * np.arccos(np.clip(abs((quat[1:] * quat[:-1]).sum(1)), 0, 1)) * fps
    start = min(len(speed), max(1, int(round(.1 * fps))))
    tail = slice(start, min(len(speed), int(round(fps))))
    later_speed = float(np.percentile(speed[tail], 95)) if len(speed[tail]) else 0.
    later_angular = float(np.percentile(angular[tail], 95)) if len(angular[tail]) else 0.
    impulse = (len(speed[tail]) >= 3 and
               ((speed[:start].max() > max(2., 8 * max(.25, later_speed))) or
                (angular[:start].max() > max(8., 8 * max(1., later_angular)))))
    checks = dict(root_speed_below_10m_s=bool(speed.max() <= 10.),
                  root_angular_speed_below_20rad_s=bool(angular.max() <= 20.),
                  no_initial_solver_impulse=not bool(impulse))
    return dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
                max_speed_m_s=float(speed.max()), max_speed_interval=int(speed.argmax()),
                max_angular_speed_rad_s=float(angular.max()),
                initial_speed_m_s=float(speed[0]), initial_angular_speed_rad_s=float(angular[0]),
                startup_max_speed_m_s=float(speed[:start].max()), later_first_second_p95_speed_m_s=later_speed,
                scope='Gross root-motion/startup checks; not a proof of dynamic feasibility')


def validate_root_motion(values, fps=30., label='reference'):
    report = root_motion_report(values, fps)
    if report['status'] != 'PASS':
        failed = [key for key, passed in report['checks'].items() if not passed]
        raise ValueError(f'{label}: unsafe reference root motion ({", ".join(failed)}); '
                         f'max speed={report["max_speed_m_s"]:.3f} m/s, '
                         f'initial angular speed={report["initial_angular_speed_rad_s"]:.3f} rad/s. '
                         'Regenerate with the first-frame IK warmup; do not resume training on this reference.')
    return report
