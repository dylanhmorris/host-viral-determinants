import numpyro as npro
import jax
import jax.numpy as jnp
from numpyro.distributions import constraints
from numpyro.distributions.util import is_prng_key
from numpyro.distributions.util import validate_sample
from numpyro.distributions.util import promote_shapes
from util import log1m_exp
from jax.scipy.special import logsumexp


class PoissonSingleHit(npro.distributions.Distribution):
    """
    Poisson Single-Hit Distribution

    This is a distribution that yields a 1
    if a Poisson random variable is non-zero
    and a zero otherwise. It occurs in virology
    because if we expose a set of cells to some
    quantity of infectious virus particles
    ("virions"), the number that succesfully enter
    a cell and replicate can be modeled as a
    Poisson distributed random variable with a mean
    related to the initial quantity of virions.
    The probability of seeing any evidence of cell
    invasion is then equal to the probability that
    this Poisson random variable is non-zero
    (i.e. at least one virion successfully
    invaded a cell).

    Parameters
    ----------
    rate : :py:class:`float`
        The rate of the Poisson random variable.
    """

    arg_constraints = {
        "rate": constraints.positive}

    support = constraints.boolean

    def __init__(self,
                 rate=0,
                 validate_args=None):

        self.rate = rate
        batch_shape = jnp.shape(self.rate)

        self.bernoulli_ = npro.distributions.Bernoulli(
            probs=1 - jnp.exp(-self.rate),
            validate_args=True)

        super().__init__(
            batch_shape=batch_shape,
            validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        """
        Parameters
        ----------
        key :
        sample_shape :
             (Default value = ())

        Returns
        -------

        """
        assert is_prng_key(key)
        return self.bernoulli_.sample(
            key,
            sample_shape=sample_shape)

    @validate_sample
    def log_prob(self, value):
        """

        Parameters
        ----------
        value :

        Returns
        -------

        """
        return self.bernoulli_.log_prob(value)


class CensoredNormal(npro.distributions.Distribution):

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive}

    support = constraints.real

    def __init__(self,
                 loc=0,
                 scale=1,
                 lower_limit=-jnp.inf,
                 upper_limit=jnp.inf,
                 validate_args=None):

        self.loc, self.scale = promote_shapes(loc, scale)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

        batch_shape = jax.lax.broadcast_shapes(jnp.shape(loc),
                                               jnp.shape(scale))
        self.normal_ = npro.distributions.Normal(
            loc=loc,
            scale=scale,
            validate_args=validate_args)
        super().__init__(
            batch_shape=batch_shape,
            validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        result = self.normal_.sample(key, sample_shape)
        return jnp.clip(result,
                        a_min=self.lower_limit,
                        a_max=self.upper_limit)

    @validate_sample
    def log_prob(self, value):
        rescaled_ulim = (self.upper_limit - self.loc) / self.scale
        rescaled_llim = (self.lower_limit - self.loc) / self.scale
        lim_val = jnp.where(
            value <= self.lower_limit,
            jax.scipy.special.log_ndtr(rescaled_llim),
            jax.scipy.special.log_ndtr(-rescaled_ulim))
        # we exploit the fact that for the
        # standard normal, P(x > a) = P(-x < a)
        # to compute the log complementary CDF
        inbounds = jnp.logical_and(
            value > self.lower_limit,
            value < self.upper_limit)
        result = jnp.where(
            inbounds,
            self.normal_.log_prob(value),
            lim_val)

        return result


class TiterWells(npro.distributions.Distribution):

    arg_constraints = {
        "log_titer_id50": constraints.real,
        "first_row_dilution": constraints.real,
        "wells_per_row": constraints.positive_integer,
        "log_dilution_factor": constraints.real,
        "log_base": constraints.positive}

    support = constraints.nonnegative_integer

    def __init__(self,
                 log_titer_id50=None,
                 first_row_dilution=None,
                 wells_per_row=None,
                 log_dilution_factor=1,
                 log_base=10,
                 validate_args=None):

        self.wells_per_row = wells_per_row

        (self.log_titer_id50,
         self.first_row_dilution,
         self.wells_per_row_shape,
         self.log_dilution_factor,
         self.log_base) = promote_shapes(log_titer_id50,
                                         first_row_dilution,
                                         wells_per_row,
                                         log_dilution_factor,
                                         log_base)
        batch_shape = jax.lax.broadcast_shapes(
            jnp.shape(self.log_titer_id50),
            jnp.shape(self.first_row_dilution),
            jnp.shape(self.wells_per_row_shape),
            jnp.shape(self.log_dilution_factor),
            jnp.shape(self.log_base))

        self.first_row_hit_rate = (
            1e-20 +
            jnp.log(2) *  # convert id50 to hit units
            jnp.exp(
                jnp.log(self.log_base) *
                (self.log_titer_id50 + self.first_row_dilution)))
        self.second_row_hit_rate = (
            1e-20 +
            jnp.log(2) *  # convert id50 to hit units
            jnp.exp(
                jnp.log(self.log_base) *
                (self.log_titer_id50 + self.first_row_dilution -
                 self.log_dilution_factor)))

        self.first_row_log_prob_miss = -self.first_row_hit_rate
        self.first_row_log_prob_hit = log1m_exp(
            self.first_row_log_prob_miss)

        self.second_row_log_prob_miss = -self.second_row_hit_rate
        self.second_row_log_prob_hit = log1m_exp(
            self.second_row_log_prob_miss)

        self.binomial_first_ = npro.distributions.Binomial(
            total_count=self.wells_per_row,
            logits=(self.first_row_log_prob_hit -
                    self.first_row_log_prob_miss))

        self.binomial_second_ = npro.distributions.Binomial(
            total_count=self.wells_per_row,
            logits=(self.second_row_log_prob_hit -
                    self.second_row_log_prob_miss))

        super().__init__(
            batch_shape=batch_shape,
            validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return (
            self.binomial_first_.sample(
                key,
                sample_shape=sample_shape) +
            self.binomial_second_.sample(
                key,
                sample_shape=sample_shape))

    @validate_sample
    def log_prob(self, value):
        first_row_counts = (
            jnp.arange(0, self.wells_per_row + 1, 1)[..., jnp.newaxis] +
            jnp.zeros_like(value[jnp.newaxis, ...]))
        second_row_counts = value - first_row_counts

        tot_probs = (
            self.binomial_first_.log_prob(first_row_counts) +
            self.binomial_second_.log_prob(second_row_counts))
        result = logsumexp(tot_probs, axis=0)
        # sum each columns (the possible ways to generate)
        # each observed titer
        return result


class WellPlate(npro.distributions.Distribution):

    arg_constraints = {
        "log_titer_id50": constraints.real,
        "row_dilutions": constraints.real,
        "wells_per_row": constraints.positive_integer,
        "log_base": constraints.positive}

    support = constraints.nonnegative_integer

    def __init__(self,
                 log_titer_id50=None,
                 row_dilutions=None,
                 wells_per_row=None,
                 log_base=10,
                 validate_args=None):

        log_titer_id50 = jnp.array(
            log_titer_id50)[..., jnp.newaxis, jnp.newaxis]

        (self.log_titer_id50,
         self.row_dilutions) = promote_shapes(log_titer_id50,
                                              row_dilutions)

        self.wells_per_row = wells_per_row
        self.log_base = log_base

        batch_shape = jax.lax.broadcast_shapes(
            jnp.shape(self.log_titer_id50),
            jnp.shape(self.row_dilutions))

        self.dilutions = jnp.tile(
            self.row_dilutions,
            self.wells_per_row
        ).reshape(
            (-1, self.wells_per_row)
        )
        self.row_hit_rates = (
            1e-20 +
            jnp.log(2) *  # convert id50 to hit units
            jnp.exp(
                jnp.log(self.log_base) *
                (self.log_titer_id50 + self.dilutions))
        )

        self.log_prob_miss = -self.row_hit_rates
        self.log_prob_hit = log1m_exp(
            self.log_prob_miss)
        self.logit_hit = self.log_prob_hit - self.log_prob_miss

        self.bernoulli_ = npro.distributions.Bernoulli(
            logits=self.logit_hit)

        super().__init__(
            batch_shape=batch_shape,
            validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return (
            self.bernoulli_.sample(
                key,
                sample_shape=sample_shape))

    @validate_sample
    def log_prob(self, value):
        return self.bernoulli_.log_prob(
            value)
