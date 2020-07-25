#!/usr/bin/env python3

import argparse
import datetime
import math
import random
import statistics
import sys
import time
from collections import namedtuple
from functools import partial
from operator import attrgetter

def bits_to_target(bits):
    size = bits >> 24
    assert size <= 0x1d

    word = bits & 0x00ffffff
    assert 0x8000 <= word <= 0x7fffff

    if size <= 3:
        return word >> (8 * (3 - size))
    else:
        return word << (8 * (size - 3))

MAX_BITS = 0x1d00ffff
MAX_TARGET = bits_to_target(MAX_BITS)

def target_to_bits(target):
    assert target > 0
    if target > MAX_TARGET:
        print('Warning: target went above maximum ({} > {})'
              .format(target, MAX_TARGET), file=sys.stderr)
        target = MAX_TARGET
    size = (target.bit_length() + 7) // 8
    mask64 = 0xffffffffffffffff
    if size <= 3:
        compact = (target & mask64) << (8 * (3 - size))
    else:
        compact = (target >> (8 * (size - 3))) & mask64

    if compact & 0x00800000:
        compact >>= 8
        size += 1

    assert compact == (compact & 0x007fffff)
    assert size < 256
    return compact | size << 24

def bits_to_work(bits):
    return (2 << 255) // (bits_to_target(bits) + 1)

def target_to_hex(target):
    h = hex(target)[2:]
    return '0' * (64 - len(h)) + h

TARGET_1 = bits_to_target(486604799)

INITIAL_BCC_BITS = 403458999
INITIAL_SWC_BITS = 402734313
INITIAL_FX = 0.18
INITIAL_TIMESTAMP = 1503430225
INITIAL_HASHRATE = 500    # In PH/s.
INITIAL_HEIGHT = 481824
INITIAL_SINGLE_WORK = bits_to_work(INITIAL_BCC_BITS)

# Steady hashrate mines the BCC chain all the time.  In PH/s.
STEADY_HASHRATE = 300

# Variable hash is split across both chains according to relative
# revenue.  If the revenue ratio for either chain is at least 15%
# higher, everything switches.  Otherwise the proportion mining the
# chain is linear between +- 15%.
VARIABLE_HASHRATE = 2000   # In PH/s.
VARIABLE_PCT = 15   # 85% to 115%
VARIABLE_WINDOW = 6  # No of blocks averaged to determine revenue ratio

# Greedy hashrate switches chain if that chain is more profitable for
# GREEDY_WINDOW BCC blocks.  It will only bother to switch if it has
# consistently been GREEDY_PCT more profitable.
GREEDY_HASHRATE = 2000     # In PH/s.
GREEDY_PCT = 1
GREEDY_WINDOW = 0

IDEAL_BLOCK_TIME = 10 * 60

State = namedtuple('State', 'height wall_time timestamp bits chainwork fx '
                   'hashrate rev_ratio greedy_frac msg')

states = []

def print_headers():
    print(', '.join(['Height', 'FX', 'Block Time', 'Unix', 'Timestamp',
                     'Difficulty (bn)', 'Implied Difficulty (bn)',
                     'Hashrate (PH/s)', 'Rev Ratio', 'Greedy?', 'Comments']))

def print_state():
    state = states[-1]
    block_time = state.timestamp - states[-2].timestamp
    t = datetime.datetime.fromtimestamp(state.timestamp)
    difficulty = TARGET_1 / bits_to_target(state.bits)
    implied_diff = TARGET_1 / ((2 << 255) / (state.hashrate * 1e15 * IDEAL_BLOCK_TIME))
    print(', '.join(['{:d}'.format(state.height),
                     '{:.8f}'.format(state.fx),
                     '{:d}'.format(block_time),
                     '{:d}'.format(state.timestamp),
                     '{:%Y-%m-%d %H:%M:%S}'.format(t),
                     '{:.2f}'.format(difficulty / 1e9),
                     '{:.2f}'.format(implied_diff / 1e9),
                     '{:.0f}'.format(state.hashrate),
                     '{:.3f}'.format(state.rev_ratio),
                     'Yes' if state.greedy_frac == 1.0 else 'No',
                     state.msg]))

def revenue_ratio(fx, BCC_target):
    '''Returns the instantaneous SWC revenue rate divided by the
    instantaneous BCC revenue rate.  A value less than 1.0 makes it
    attractive to mine BCC.  Greater than 1.0, SWC.'''
    SWC_fees = 0.25 + 2.0 * random.random()
    SWC_revenue = 12.5 + SWC_fees
    SWC_target = bits_to_target(INITIAL_SWC_BITS)

    BCC_fees = 0.2 * random.random()
    BCC_revenue = (12.5 + BCC_fees) * fx

    SWC_difficulty_ratio = BCC_target / SWC_target
    return SWC_revenue / SWC_difficulty_ratio / BCC_revenue

def median_time_past(states):
    times = [state.timestamp for state in states]
    return sorted(times)[len(times) // 2]

def next_bits_k(msg, mtp_window, high_barrier, target_raise_frac,
                low_barrier, target_drop_frac, fast_blocks_pct):
    # Calculate N-block MTP diff
    MTP_0 = median_time_past(states[-11:])
    MTP_N = median_time_past(states[-11-mtp_window:-mtp_window])
    MTP_diff = MTP_0 - MTP_N
    bits = states[-1].bits
    target = bits_to_target(bits)

    # Long term block production time stabiliser
    t = states[-1].timestamp - states[-2017].timestamp
    if t < IDEAL_BLOCK_TIME * 2016 * fast_blocks_pct // 100:
        msg.append("2016 block time difficulty raise")
        target -= target // target_drop_frac

    if MTP_diff > high_barrier:
        target += target // target_raise_frac
        msg.append("Difficulty drop {}".format(MTP_diff))
    elif MTP_diff < low_barrier:
        target -= target // target_drop_frac
        msg.append("Difficulty raise {}".format(MTP_diff))
    else:
        msg.append("Difficulty held {}".format(MTP_diff))

    return target_to_bits(target)

def suitable_block_index(index):
    assert index >= 3
    indices = [index - 2, index - 1, index]

    if states[indices[0]].timestamp > states[indices[2]].timestamp:
        indices[0], indices[2] = indices[2], indices[0]

    if states[indices[0]].timestamp > states[indices[1]].timestamp:
        indices[0], indices[1] = indices[1], indices[0]

    if states[indices[1]].timestamp > states[indices[2]].timestamp:
        indices[1], indices[2] = indices[2], indices[1]

    return indices[1]

def compute_index_fast(index_last):
    for candidate in range(index_last - 3, 0, -1):
        index_fast = suitable_block_index(candidate)
        if index_last - index_fast < 5:
            continue
        if (states[index_last].timestamp - states[index_fast].timestamp
            >= 13 * IDEAL_BLOCK_TIME):
            return index_fast
    raise AssertionError('should not happen')

def compute_target(first_index, last_index):
    work = states[last_index].chainwork - states[first_index].chainwork
    work *= IDEAL_BLOCK_TIME
    work //= states[last_index].timestamp - states[first_index].timestamp
    return (2 << 255) // work - 1

def next_bits_d(msg):
    N = len(states) - 1
    index_last = suitable_block_index(N)
    index_first = suitable_block_index(N - 2016)
    interval_target = compute_target(index_first, index_last)
    index_fast = compute_index_fast(index_last)
    fast_target = compute_target(index_fast, index_last)

    next_target = interval_target
    if (fast_target < interval_target - (interval_target >> 2) or
        fast_target > interval_target + (interval_target >> 2)):
        msg.append("fast target")
        next_target = fast_target
    else:
        msg.append("interval target")

    prev_target = bits_to_target(states[-1].bits)
    min_target = prev_target - (prev_target >> 3)
    if next_target < min_target:
        msg.append("min target")
        return target_to_bits(min_target)

    max_target = prev_target + (prev_target >> 3)
    if next_target > max_target:
        msg.append("max target")
        return target_to_bits(max_target)

    return target_to_bits(next_target)

def compute_cw_target(block_count):
    N = len(states) - 1
    last = suitable_block_index(N)
    first = suitable_block_index(N - block_count)
    timespan = states[last].timestamp - states[first].timestamp
    timespan = max(block_count * IDEAL_BLOCK_TIME // 2, min(block_count * 2 * IDEAL_BLOCK_TIME, timespan))
    work = (states[last].chainwork - states[first].chainwork) * IDEAL_BLOCK_TIME // timespan
    return (2 << 255) // work - 1

def next_bits_sha(msg):
    primes = [73, 79, 83, 89, 97,
              101, 103, 107, 109, 113, 127,
              131, 137, 139, 149, 151]

    # The timestamp % len(primes) is a proxy for previous
    # block SHAx2 % len(primes), but that data is not available
    # in this simulation
    prime = primes[states[-1].timestamp % len(primes)]
    
    interval_target = compute_cw_target(prime)
    return target_to_bits(interval_target)

def next_bits_wave(msg, avgcount):
    primes = [ # 73,  79,  83,  89,  97, 101,
              103, 107, 109, 113, 127, 131, 137, 139,
              149, 151, 157, 163, 167, 173, 179, 181,
              191, 193, 197, 199, 211, 223, 227, 229 ]
    assert avgcount <= len(primes) and avgcount >= 0, "invalid samplesize"

    seed = states[-1].timestamp % (len(primes) // avgcount) + 1
    total_work = 0
    total_timespan = 0
    random.seed(states[-1].timestamp)

    for i in range(1, avgcount):
        prime = random.randrange( 0, len(primes))
        #prime = random.randrange( (len(primes) // avgcount) * i, (len(primes) // avgcount) * (i+1))
        work, timespan = wave_data(i, primes[prime])
        interval_target = (1 << 256) // (work * 600 // timespan) - 1
        total_work += work
        total_timespan += timespan

    normalized_work = (total_work * 600) // total_timespan
    target = (1 << 256) // normalized_work - 1
    #last_target = bits_to_target(states[-1].bits)
    #if target > (last_target << 1):
    #    target = last_target << 1
    #elif target < (last_target >> 1):
    #    target = last_target >> 1

    return target_to_bits(target)

def next_bits_wave_random(msg, avgcount):
    import random

    random.seed(states[-1].timestamp)
    total_work = 0
    total_timespan = 0
    element = 1
    for i in range(1, avgcount):
        element = random.randrange(73, 201)
        work, timespan = wave_data(i, element)
        total_work += work
        total_timespan += timespan

    normalized_work = (total_work * 600) // total_timespan
    target = (1 << 256) // normalized_work - 1

    return target_to_bits(target)

def next_bits_wtema(msg, alpha_recip):
    # This algorithm is weighted-target exponential moving average.
    # Target is calculated based on inter-block times weighted by a
    # progressively decreasing factor for past inter-block times,
    # according to the parameter alpha.  If the single_block_target SBT is
    # calculated as:
    #    SBT = prior_target * block_time / ideal_block_time
    # then:
    #    next_target = SBT * α + prior_target * (1 - α)
    # Substituting and factorizing:
    #    next_target = prior_target * α / ideal_block_time
    #                  * (block_time + (1 / α - 1) * ideal_block_time)
    # We use the reciprocal of alpha as an integer to avoid floating
    # point arithmetic.  Doing so the above formula maintains precision and
    # avoids overflows wih large targets in regtest
    timespan = states[-1].timestamp - states[-2].timestamp
    block_time = max(IDEAL_BLOCK_TIME // 2, min(2 * IDEAL_BLOCK_TIME, timespan))

    prior_target = bits_to_target(states[-1].bits)
    next_target = prior_target // (IDEAL_BLOCK_TIME * alpha_recip)
    next_target *= block_time + IDEAL_BLOCK_TIME * (alpha_recip - 1)
    return target_to_bits(next_target)


def next_bits_wtema_test(msg, alpha_recip):
    # This algorithm is weighted-target exponential moving average.
    # Target is calculated based on inter-block times weighted by a
    # progressively decreasing factor for past inter-block times,
    # according to the parameter alpha.  If the single_block_target SBT is
    # calculated as:
    #    SBT = prior_target * block_time / ideal_block_time
    # then:
    #    next_target = SBT * α + prior_target * (1 - α)
    # Substituting and factorizing:
    #    next_target = prior_target * α / ideal_block_time
    #                  * (block_time + (1 / α - 1) * ideal_block_time)
    # We use the reciprocal of alpha as an integer to avoid floating
    # point arithmetic.  Doing so the above formula maintains precision and
    # avoids overflows wih large targets in regtest


    block_count = 1
    ideal_time = IDEAL_BLOCK_TIME * block_count
    start = len(states) - 1
    first, last  = suitable_block_index(start - block_count), suitable_block_index(start)
    timespan = states[last].timestamp - states[first].timestamp
    block_time = max(block_count * ideal_time // 2, min(block_count * 2 * ideal_time, timespan))

    prior_target = bits_to_target(states[-1].bits)
    if states[-1].height % 2000 == 0:
        prior_target = bits_to_target(next_bits_cw(msg, 2000))
    next_target = prior_target // (ideal_time * alpha_recip)
    next_target *= block_time + ideal_time * (alpha_recip - 1)
    return target_to_bits(next_target)


def wave_data(start, block_count):
    start = len(states) - 1
    first, last  = suitable_block_index(start - block_count), suitable_block_index(start)
    timespan = states[last].timestamp - states[first].timestamp
    # Cap limit if something weird happens for 3 blocks in a row
    timespan = max(block_count * IDEAL_BLOCK_TIME // 2, min(block_count * 2 * IDEAL_BLOCK_TIME, timespan))
    work = (states[last].chainwork - states[first].chainwork)
    return work, timespan


def next_bits_avg(msg, algo, avgcount):
    primes = [ 13,  17,  19,  23,  29,  31,  37,  41,
               43,  47,  73,  79,  83,  89,  97, 101,
              103, 107, 109, 113, 127, 131, 137, 139,
              149, 151, 157, 163, 167, 173, 179, 181,
              191, 193, 197, 199, 211, 223, 227, 229,
              233, 239, 241, 251, 257, 263, 269, 271 ]
    assert avgcount <= len(primes) and avgcount >= 0, "invalid samplesize"

    interval_target = 0
    for i in range(0, avgcount):
        interval_target += algo(primes[i]) // avgcount

    return target_to_bits(interval_target)



def next_bits_cw(msg, block_count):
    interval_target = compute_cw_target(block_count)
    return target_to_bits(interval_target)

def next_bits_wt(msg, block_count):
    first, last  = -1-block_count, -1
    timespan = 0
    prior_timestamp = states[first].timestamp
    for i in range(first + 1, last + 1):
        target_i = bits_to_target(states[i].bits)

        # Prevent negative time_i values
        timestamp = max(states[i].timestamp, prior_timestamp)
        time_i = timestamp - prior_timestamp
        prior_timestamp = timestamp
        adj_time_i = time_i * target_i # Difficulty weight
        timespan += adj_time_i * (i - first) # Recency weight

    timespan = timespan * 2 // (block_count + 1) # Normalize recency weight
    target = timespan // (IDEAL_BLOCK_TIME * block_count)
    return target_to_bits(target)

def next_bits_wt_compare(msg, block_count):
    with open("current_state.csv", 'w') as fh:
        for s in states:
            fh.write("%s,%s,%s\n" % (s.height, s.bits, s.timestamp))

    from subprocess import Popen, PIPE

    process = Popen(["./cashwork"], stdout=PIPE)
    (next_bits, err) = process.communicate()
    exit_code = process.wait()

    next_bits = int(next_bits.decode())
    next_bits_py = next_bits_wt(msg, block_count)
    if next_bits != next_bits_py:
        print("ERROR: Bits don't match. External %s, local %s" % (next_bits, next_bits_py))
        assert(next_bits == next_bits_py)
    return next_bits


def next_bits_dgw3(msg, block_count):
    ''' Dark Gravity Wave v3 from Dash '''
    block_reading = -1 # dito
    counted_blocks = 0
    last_block_time = 0
    actual_time_span = 0
    past_difficulty_avg = 0
    past_difficulty_avg_prev = 0
    i = 1
    while states[block_reading].height > 0:
        if i > block_count:
            break
        counted_blocks += 1
        if counted_blocks <= block_count:
            if counted_blocks == 1:
                past_difficulty_avg = bits_to_target(states[block_reading].bits)
            else:
                past_difficulty_avg = ((past_difficulty_avg_prev * counted_blocks) + bits_to_target(states[block_reading].bits)) // ( counted_blocks + 1 )
        past_difficulty_avg_prev = past_difficulty_avg
        if last_block_time > 0:
            diff = last_block_time - states[block_reading].timestamp
            actual_time_span += diff
        last_block_time = states[block_reading].timestamp
        block_reading -= 1
        i += 1
    target_time_span = counted_blocks * IDEAL_BLOCK_TIME
    target = past_difficulty_avg
    if actual_time_span < (target_time_span // 3):
        actual_time_span = target_time_span // 3
    if actual_time_span > (target_time_span * 3):
        actual_time_span = target_time_span * 3
    target = target // target_time_span
    target *= actual_time_span
    if target > MAX_TARGET:
        return MAX_BITS
    else:
        return target_to_bits(int(target))

def next_bits_m2(msg, window_1, window_2):
    interval_target = compute_target(-1 - window_1, -1)
    interval_target += compute_target(-2 - window_2, -2)
    return target_to_bits(interval_target >> 1)

def next_bits_m4(msg, window_1, window_2, window_3, window_4):
    interval_target = compute_target(-1 - window_1, -1)
    interval_target += compute_target(-2 - window_2, -2)
    interval_target += compute_target(-3 - window_3, -3)
    interval_target += compute_target(-4 - window_4, -4)
    return target_to_bits(interval_target >> 2)

def next_bits_ema(msg, window):
    block_time          = states[-1].timestamp - states[-2].timestamp
    block_time          = max(IDEAL_BLOCK_TIME / 100, min(100 * IDEAL_BLOCK_TIME, block_time))          # Crudely dodge problems from ~0/negative/huge block times
    old_hashrate_est    = TARGET_1 / bits_to_target(states[-1].bits)                                    # "Hashrate estimate" - aka difficulty!
    block_weight        = 1 - math.exp(-block_time / window)                                            # Weight of last block_time seconds, according to exp moving avg
    block_hashrate_est  = (IDEAL_BLOCK_TIME / block_time) * old_hashrate_est                            # Eg, if a block takes 2 min instead of 10, we est hashrate was ~5x higher than predicted
    new_hashrate_est    = (1 - block_weight) * old_hashrate_est + block_weight * block_hashrate_est     # Simple weighted avg of old hashrate est, + block's adjusted hashrate est
    new_target          = round(TARGET_1 / new_hashrate_est)
    return target_to_bits(new_target)

def next_bits_asert_discrete(msg, window, granularity = 144):
    """Another exponential-decay-based algo that uses integer math instead of exponentiation.  As with asert, we increase difficulty by a fixed amount each time a block is found,
    and decrease it steadily for the passage of time between found blocks.  But here both adjustments are done in integer math, using the principles that:
    1. We can discretize "Decrease difficulty steadily by a factor of 1/e over (say) 1 day", into "Decrease difficulty by exactly 1/e^(1/100) for each 1/100 of a day."
    2. We can closely approximate "difficulty * 1/e^(1/100)", by "(difficulty * 4252231657) >> 32".  (Really, just "difficulty * 99 // 100" would probably do the job too.)
    The "window" param is meant to invoke the fixed time window of a simple moving average, but here we give it the standard equivalent EMA interpretation: the window is "how old a
    block has to be (in seconds) for us to discount its weight by a factor of e."  So, instead of 86400 (1 day) meaning "average the block times over the last day," here it means 
    "in our weighted avg of block times, give a day-old block 1/e the weight of the latest block."  This results in comparable responsiveness to a fixed 1-day window.
    Given this framework, we update the target as follows:
    1. Decrease the difficulty (ie, increase the target) by the constant factor we always increase it by for each new block.  (See the ASERT algo for an explanation of this.)
    2. Adjust the difficulty based on the passage of time (typically increase it, except in the unusual case where this block's timestamp is before the previous block's):
       a) Specify a granularity - the number of segments we'll divide the time window into.  Eg, window = 86400 and granularity = 100, means segment = 864.
       b) Figure out which numbered segment (since genesis) the previous block's timestamp fell into, and which segment the current block falls into.
       c) The difference between the segment numbers of the two blocks tells us how many discrete "per-segment difficulty adjustments" we need to make.  Eg, if the current block
          lands in the segment 3 segments after the one the previous block did, we need to decrease difficulty by three "segment adjustments".
       d) Theoretically, if granularity = 100, we should multiply difficulty by e**(-1/100) for each segment.  We can approximate each adjustment by an int-math multiplication, 
          using the e^(-1/100)*(2**32) in TIME_SEGMENT_DIFFICULTY_DECREASE_FACTOR above.  This means our algo can only time-adjust difficulty by multiples of 1% - but that's OK."""

    FACTOR_SCALING_BITS = 32
    # These factors are scaled by 2**FACTOR_SCALING_BITS.  Eg, (x * 4252231657) >> 32, is approximately x * e^(-1/100).
    TIME_SEGMENT_DIFFICULTY_DECREASE_FACTOR = {
        100: 4252231657,                    # If current block's timestamp is one segment later than previous block's, then multiply difficulty by this number and divide by 2**32.
    }
    BLOCK_DIFFICULTY_INCREASE_FACTOR = {
        144: 4324897261,                    # If window = 144 * IDEAL_BLOCK_TIME, then every time a block is found, multiply difficulty by this number and divide by 2**32.
    }

    old_segment_number = states[-2].timestamp // (window // granularity)
    new_segment_number = states[-1].timestamp // (window // granularity)
    old_target = bits_to_target(states[-1].bits)

    # We divide by the factors here, rather than multiply, because we're actually adjusting target, not difficulty:
    new_target = (old_target << FACTOR_SCALING_BITS) // BLOCK_DIFFICULTY_INCREASE_FACTOR[window // IDEAL_BLOCK_TIME]
    if new_segment_number > old_segment_number:
        # Doing this in a simple for loop means that a pathological block time (far in past or future) could make this very slow.  I'm not sure such blocks ever actually occur,
        # but if this were a concern we could speed this up via https://en.wikipedia.org/wiki/Exponentiation_by_squaring.  Eg, if the two block times are 20 segments apart, a naive
        # loop does 20 multiplications/divisions (plus bit-shifts), whereas exp by squaring could do it in 6 as follows:
        #     e**(2/100)  = (e**(1/100))**2
        #     e**(4/100)  = (e**(2/100))**2
        #     e**(8/100)  = (e**(4/100))**2
        #     e**(16/100) = (e**(8/100))**2
        #     e**(20/100) = e**(16/100) * e**(4/100)
        #     new_target = old_target * e**(20/100)
        # This saving gets significant for large numbers (log vs linear): if the blocks were 1,000,000 segments apart, this would mean 26 multiplications rather than 1,000,000.
        for _ in range(old_segment_number, new_segment_number):
            new_target = (new_target << FACTOR_SCALING_BITS) // TIME_SEGMENT_DIFFICULTY_DECREASE_FACTOR[granularity]
    elif new_segment_number < old_segment_number:                       # If the new block's timestamp is weirdly before the old one's, OK then, *increase* difficulty accordingly
        for _ in range(new_segment_number, old_segment_number):
            new_target = (new_target * TIME_SEGMENT_DIFFICULTY_DECREASE_FACTOR[granularity]) >> FACTOR_SCALING_BITS

    return target_to_bits(new_target)

def block_time(mean_time):
    # Sample the exponential distn
    sample = random.random()
    lmbda = 1 / mean_time
    return math.log(1 - sample) / -lmbda

def next_fx_random(r):
    return states[-1].fx * (1.0 + (r - 0.5) / 200)

def next_fx_none(r):
    return states[-1].fx

def next_fx_ramp(r):
    return states[-1].fx * 1.00017149454

def next_step(algo, scenario, fx_jump_factor):
    # First figure out our hashrate
    msg = []
    high = 1.0 + VARIABLE_PCT / 100
    scale_fac = 50 / VARIABLE_PCT
    N = VARIABLE_WINDOW
    mean_rev_ratio = sum(state.rev_ratio for state in states[-N:]) / N
    var_fraction = max(0, min(1, (high - mean_rev_ratio) * scale_fac))
    if ((scenario.pump_144_threshold > 0) and
        (states[-1-144+5].timestamp - states[-1-144].timestamp > scenario.pump_144_threshold)):
        var_fraction = max(var_fraction, .25)

    # Calculate our dynamic difficulty
    bits = algo.next_bits(msg, **algo.params)
    target = bits_to_target(bits)

    # Get a new FX rate
    rand = random.random()
    fx = scenario.next_fx(rand, **scenario.params)
    if fx_jump_factor != 1.0:
        msg.append('FX jumped by factor {:.2f}'.format(fx_jump_factor))
        fx *= fx_jump_factor

    N = GREEDY_WINDOW
    rev_ratio = revenue_ratio(fx, target)
    if N > 0:
        gready_rev_ratio = sum(state.rev_ratio for state in states[-N:]) / N
    
    greedy_frac = states[-1].greedy_frac
    if mean_rev_ratio >= 1 + GREEDY_PCT / 100:
        if greedy_frac != 0.0:
            msg.append("Greedy miners left")
        greedy_frac = 0.0
    elif mean_rev_ratio <= 1 - GREEDY_PCT / 100:
        if greedy_frac != 1.0:
            msg.append("Greedy miners joined")
        greedy_frac = 1.0

    hashrate = (STEADY_HASHRATE + scenario.dr_hashrate
                + VARIABLE_HASHRATE * var_fraction
                + GREEDY_HASHRATE * greedy_frac)

    # See how long we take to mine a block
    mean_hashes = pow(2, 256) // target
    mean_time = mean_hashes / (hashrate * 1e15)
    time = int(block_time(mean_time) + 0.5)
    wall_time = states[-1].wall_time + time
    # Did the difficulty ramp hashrate get the block?
    if random.random() < (abs(scenario.dr_hashrate) / hashrate):
        if (scenario.dr_hashrate > 0):
            timestamp = median_time_past(states[-11:]) + 1
        else:
            timestamp = wall_time + 2 * 60 * 60
    else:
        timestamp = wall_time

    chainwork = states[-1].chainwork + bits_to_work(bits)

    # add a state
    states.append(State(states[-1].height + 1, wall_time, timestamp,
                        bits, chainwork, fx, hashrate, rev_ratio,
                        greedy_frac, ' / '.join(msg)))

Algo = namedtuple('Algo', 'next_bits params')

Algos = {
    'k-1' : Algo(next_bits_k, {
        'mtp_window': 6,
        'high_barrier': 60 * 128,
        'target_raise_frac': 64,   # Reduce difficulty ~ 1.6%
        'low_barrier': 60 * 30,
        'target_drop_frac': 256,   # Raise difficulty ~ 0.4%
        'fast_blocks_pct': 95,
    }),
    'k-2' : Algo(next_bits_k, {
        'mtp_window': 4,
        'high_barrier': 60 * 55,
        'target_raise_frac': 100,   # Reduce difficulty ~ 1.0%
        'low_barrier': 60 * 36,
        'target_drop_frac': 256,   # Raise difficulty ~ 0.4%
        'fast_blocks_pct': 95,
    }),
    'd-1' : Algo(next_bits_d, {}),
    'cw-72' : Algo(next_bits_cw, {
        'block_count': 72,
    }),
    'cw-108' : Algo(next_bits_cw, {
        'block_count': 108,
    }),
    'cw-144' : Algo(next_bits_cw, {
        'block_count': 144,
    }),
    'cw-sha-16' : Algo(next_bits_sha, {}),
    'cw-avg-16' : Algo(next_bits_avg, {
        'avgcount': 16,
        'algo': compute_cw_target
    }),
    'cw-180' : Algo(next_bits_cw, {
        'block_count': 180,
    }),
    'ksch-5' : Algo(next_bits_wave, {
        'avgcount': 5,
    }),
    'ksch-7' : Algo(next_bits_wave, {
        'avgcount': 7,
    }),
    'r-ksch-5' : Algo(next_bits_wave_random, {
        'avgcount': 5,
    }),
    'wt-144' : Algo(next_bits_wt, {
        'block_count': 144
    }),
    'dgw3-24' : Algo(next_bits_dgw3, { # 24-blocks, like Dash
        'block_count': 24,
    }),
    'dgw3-144' : Algo(next_bits_dgw3, { # 1 full day
        'block_count': 144,
    }),
    'meng-1' : Algo(next_bits_m2, { # mengerian_algo_1
        'window_1': 71,
        'window_2': 137,
    }),
    'meng-2' : Algo(next_bits_m4, { # mengerian_algo_2
        'window_1': 13,
        'window_2': 37,
        'window_3': 71,
        'window_4': 137,
    }),
    # runs wt-144 in external program, compares with python implementation.
    'wt-144-compare' : Algo(next_bits_wt_compare, {
        'block_count': 144
    }),
    'ema-30min' : Algo(next_bits_ema, { # Exponential moving avg
        'window': 30 * 60,
    }),
    'ema-3h' : Algo(next_bits_ema, {
        'window': 3 * 60 * 60,
    }),
    'ema-1d' : Algo(next_bits_ema, {
        'window': 24 * 60 * 60,
    }),
    'wtema-72' : Algo(next_bits_wtema, {
        'alpha_recip': 104, # floor(1/(1 - pow(.5, 1.0/72))), # half-life = 72
    }),
    'wtema-test-72' : Algo(next_bits_wtema_test, {
        'alpha_recip': 104, # floor(1/(1 - pow(.5, 1.0/72))), # half-life = 72
    }),
    'asertd-144' : Algo(next_bits_asert_discrete, {
        'window': (IDEAL_BLOCK_TIME * 144),
        'granularity': 100,
    }),
}

Scenario = namedtuple('Scenario', 'next_fx params, dr_hashrate, pump_144_threshold')

Scenarios = {
    'default' : Scenario(next_fx_random, {}, 0, 0),
    'none': Scenario(next_fx_none, {}, 0, 0),
    'fxramp' : Scenario(next_fx_ramp, {}, 0, 0),
    # Difficulty rampers with given PH/s
    'dr50' : Scenario(next_fx_random, {}, 50, 0),
    'dr75' : Scenario(next_fx_random, {}, 75, 0),
    'dr100' : Scenario(next_fx_random, {}, 100, 0),
    'pump-osc' : Scenario(next_fx_ramp, {}, 0, 8000),
    'ft100' : Scenario(next_fx_random, {}, -100, 0),
}

def run_one_simul(algo, scenario, print_it):
    states.clear()

    # Initial state is afer 2020 steady prefix blocks
    N = 2020
    for n in range(-N, 0):
        state = State(INITIAL_HEIGHT + n, INITIAL_TIMESTAMP + n * IDEAL_BLOCK_TIME,
                      INITIAL_TIMESTAMP + n * IDEAL_BLOCK_TIME,
                      INITIAL_BCC_BITS, INITIAL_SINGLE_WORK * (n + N + 1),
                      INITIAL_FX, INITIAL_HASHRATE, 1.0, False, '')
        states.append(state)

    # Add 10 randomly-timed FX jumps (up or down 10 and 15 percent) to
    # see how algos recalibrate
    fx_jumps = {}
    factor_choices = [0.85, 0.9, 1.1, 1.15]
    for n in range(10):
        fx_jumps[random.randrange(10000)] = random.choice(factor_choices)

    # Run the simulation
    if print_it:
        print_headers()
    for n in range(10000):
        fx_jump_factor = fx_jumps.get(n, 1.0)
        next_step(algo, scenario, fx_jump_factor)
        if print_it:
            print_state()

    # Drop the prefix blocks to be left with the simulation blocks
    simul = states[N:]

    block_times = [simul[n + 1].timestamp - simul[n].timestamp
                   for n in range(len(simul) - 1)]
    return block_times


def main():
    '''Outputs CSV data to stdout.   Final stats to stderr.'''

    parser = argparse.ArgumentParser('Run a mining simulation')
    parser.add_argument('-a', '--algo', metavar='algo', type=str,
                        choices = list(Algos.keys()),
                        default = 'k-1', help='algorithm choice')
    parser.add_argument('-s', '--scenario', metavar='scenario', type=str,
                        choices = list(Scenarios.keys()),
                        default = 'default', help='scenario choice')
    parser.add_argument('-r', '--seed', metavar='seed', type=int,
                        default = None, help='random seed')
    parser.add_argument('-n', '--count', metavar='count', type=int,
                        default = 1, help='count of simuls to run')
    args = parser.parse_args()

    count = max(1, args.count)
    algo = Algos.get(args.algo)
    scenario = Scenarios.get(args.scenario)
    seed = int(time.time()) if args.seed is None else args.seed

    to_stderr = partial(print, file=sys.stderr)
    to_stderr("Starting seed {} for {} simuls".format(seed, count))

    means = []
    std_devs = []
    medians = []
    maxs = []
    for loop in range(count):
        random.seed(seed)
        seed += 1
        block_times = run_one_simul(algo, scenario, count == 1)
        means.append(statistics.mean(block_times))
        std_devs.append(statistics.stdev(block_times))
        medians.append(sorted(block_times)[len(block_times) // 2])
        maxs.append(max(block_times))

    def stats(text, values):
        if count == 1:
            to_stderr('{} {}s'.format(text, values[0]))
        else:
            to_stderr('{}(s) Range {:0.1f}-{:0.1f} Mean {:0.1f} '
                      'Std Dev {:0.1f} Median {:0.1f}'
                      .format(text, min(values), max(values),
                              statistics.mean(values),
                              statistics.stdev(values),
                              sorted(values)[len(values) // 2]))

    stats("Mean   block time", means)
    stats("StdDev block time", std_devs)
    stats("Median block time", medians)
    stats("Max    block time", maxs)

if __name__ == '__main__':
    main()
