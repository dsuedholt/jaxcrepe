import scipy
import jax.numpy as jnp

import jaxcrepe


###############################################################################
# Pitch unit conversions
###############################################################################


def bins_to_cents(bins):
    """Converts pitch bins to cents"""
    cents = jaxcrepe.CENTS_PER_BIN * bins + 1997.3794084376191

    # Trade quantization error for noise
    return dither(cents)


def bins_to_frequency(bins):
    """Converts pitch bins to frequency in Hz"""
    return cents_to_frequency(bins_to_cents(bins))


def cents_to_bins(cents, quantize_fn=jnp.floor):
    """Converts cents to pitch bins"""
    bins = (cents - 1997.3794084376191) / jaxcrepe.CENTS_PER_BIN
    return quantize_fn(bins).astype(int)


def cents_to_frequency(cents):
    """Converts cents to frequency in Hz"""
    return 10 * 2 ** (cents / 1200)


def frequency_to_bins(frequency, quantize_fn=jnp.floor):
    """Convert frequency in Hz to pitch bins"""
    return cents_to_bins(frequency_to_cents(frequency), quantize_fn)


def frequency_to_cents(frequency):
    """Convert frequency in Hz to cents"""
    return 1200 * jnp.log2(frequency / 10.)


###############################################################################
# Utilities
###############################################################################


def dither(cents):
    """Dither the predicted pitch in cents to remove quantization error"""
    noise = scipy.stats.triang.rvs(c=0.5,
                                   loc=-jaxcrepe.CENTS_PER_BIN,
                                   scale=2 * jaxcrepe.CENTS_PER_BIN,
                                   size=cents.size)
    return cents + noise
