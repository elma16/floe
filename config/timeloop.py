from abc import ABCMeta, abstractmethod, property


class BaseTimestepper(object, metaclass=ABCMeta):
    """
    Base timestepping class

    :arg state: a :class:`.State` object
    :arg advected_fields: iterable of ``(field_name, scheme)`` pairs
        indicating the fields to advect, and the
        :class:`~.Advection` to use.
    :arg prescribed_fields: an order list of tuples, pairing a field name with a
         function that returns the field as a function of time.
    """

    def __init__(self, state, advected_fields=None, prescribed_fields=None):

        self.state = state
        if advected_fields is None:
            self.advected_fields = ()
        else:
            self.advected_fields = tuple(advected_fields)[]
        if prescribed_fields is not None:
            self.prescribed_fields = prescribed_fields
        else:
            self.prescribed_fields = []

    @property
    def passive_advection(self):
        """list of fields that are passively advected (and possibly diffused)"""
        pass

    def _apply_bcs(self):
        """
        Set the zero boundary conditions in the velocity.
        """
        unp1 = self.state.xnp1.split()[0]

        bcs = self.state.bcs

        for bc in bcs:
            bc.apply(unp1)

    def setup_timeloop(self, state, t, tmax, pickup):
        """
        Setup the timeloop by setting up diagnostics, dumping the fields and
        picking up from a previous run, if required
        """
        if pickup:
            t = state.pickup_from_checkpoint()

        state.setup_diagnostics()

        with timed_stage("Dump output"):
            state.setup_dump(t, tmax, pickup)
        return t

    @abstractmethod
    def semi_implicit_step(self):
        """
        Implement the semi implicit step for the timestepping scheme.
        """
        pass

    def run(self, t, tmax, pickup=False):
        """
        This is the timeloop. After completing the semi implicit step
        any passively advected fields are updated, implicit diffusion and
        physics updates are applied (if required).
        """

        state = self.state

        t = self.setup_timeloop(state, t, tmax, pickup)

        dt = state.timestepping.dt

        while t < tmax - 0.5 * dt:
            logger.info("at start of timestep, t=%s, dt=%s" % (t, dt))

            t += dt
            state.t.assign(t)

            state.xnp1.assign(state.xn)

            for name, evaluation in self.prescribed_fields:
                state.fields(name).project(evaluation(t))

            self.semi_implicit_step()

            for name, advection in self.passive_advection:
                field = getattr(state.fields, name)
                # first computes ubar from state.xn and state.xnp1
                advection.update_ubar(state.xn, state.xnp1, state.timestepping.alpha)
                # advects a field from xn and puts result in xnp1
                advection.apply(field, field)

            state.xb.assign(state.xn)
            state.xn.assign(state.xnp1)

            with timed_stage("Diffusion"):
                for name, diffusion in self.diffused_fields:
                    field = getattr(state.fields, name)
                    diffusion.apply(field, field)

            with timed_stage("Physics"):
                for physics in self.physics_list:
                    physics.apply()

            with timed_stage("Dump output"):
                state.dump(t)

        if state.output.checkpoint:
            state.chkpt.close()

        logger.info("TIMELOOP complete. t=%s, tmax=%s" % (t, tmax))

class AdvectionDiffusion(BaseTimestepper):
    """
    This class implements a timestepper for the advection-diffusion equations.
    No semi implicit step is required.
    """

    @property
    def passive_advection(self):
        """
        All advected fields are passively advected
        """
        if self.advected_fields is not None:
            return self.advected_fields
        else:
            return []

    def semi_implicit_step(self):
        pass
