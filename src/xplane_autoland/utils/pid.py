class PID:
    '''
    Simple class for scalar PID control
    '''
    def __init__(self, tau, kp, ki, kd, integral_max=float('inf')):
        self._tau = tau
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._integral_max = integral_max

        self._prev_error = 0.0
        self._integral   = 0.0

    def __call__(self, error):
        self._integral = min(self._integral + error * self._tau,
                             self._integral_max)
        deriv = (error - self._prev_error) / self._tau

        self._prev_error = error

        p  = self._kp * error
        i  = self._ki * self._integral
        d  = self._kd * deriv

        return p + i + d


