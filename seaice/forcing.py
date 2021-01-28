class Forcing(object):
    def __init__(self):
        ocean_curr = as_vector([0.1 * (2 * y - length) / length,
                                -0.1 * (length - 2 * x) / length])

        # initalise geo_wind
        t0 = Constant(0)

        geo_wind = as_vector(
            [5 + (sin(2 * pi * t0 / self.timescale) - 3) * sin(2 * pi * x / length) * sin(
                2 * pi * y / length),
             5 + (sin(2 * pi * t0 / self.timescale) - 3) * sin(2 * pi * y / length) * sin(
                 2 * pi * x / length)])

        # Forcing
        lm -= self.timestep * inner(
            params.rho * hh * params.cor * as_vector([uh[1] - ocean_curr[1], ocean_curr[0]
                                                      - uh[0]]), p) * dx
        # Forcing
        lm += self.timestep * inner(
            params.rho_a * params.C_a * dot(geo_wind, geo_wind) * geo_wind + params.rho_w * params.C_w * sqrt(
                dot(uh - ocean_curr, uh - ocean_curr)) * (
                    ocean_curr - uh), p) * dx