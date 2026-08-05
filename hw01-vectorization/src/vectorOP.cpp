#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // Vectorized clamped exponentiation with per-lane masking.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
      // build the active mask
    int width; 
    if (i+VECTOR_WIDTH > N) 
      width = N - i;
    else 
      width = VECTOR_WIDTH;

    __pp_mask activeMask = _pp_init_ones(width);

    // declare the vector registers 
    __pp_vec_float x;         //  x = values
    __pp_vec_int y;           //  y = exponents
    __pp_vec_float result;    // evaluated result
    __pp_vec_int count;       // loop counter

    // initialize constant registers
    __pp_vec_int zero_i = _pp_vset_int(0);
    __pp_vec_int one_i = _pp_vset_int(1);
    __pp_vec_float clamp_val = _pp_vset_float(9.999999f);

    // load data into vector registers
    _pp_vload_float(x, values + i, activeMask);
    _pp_vload_int(y, exponents + i, activeMask);

    // initialize result to 1.0 (for the case y == 0)
    _pp_vset_float(result, 1.f, activeMask);

    // to eval the case of y == 0
    __pp_mask expIsZeroMask;
    _pp_veq_int(expIsZeroMask, y, zero_i, activeMask); 
    _pp_vset_float(result, 1.f, expIsZeroMask);        // set result to 1.0 where y == 0

    // to eval the case of y > 0
    __pp_mask expIsNotZeroMask = _pp_mask_not(expIsZeroMask);
    expIsNotZeroMask = _pp_mask_and(expIsNotZeroMask, activeMask); // to prevent out-of-bound access

    _pp_vmove_float(result, x, expIsNotZeroMask); // set result to x where y != 0
    _pp_vsub_int(count, y, one_i, expIsNotZeroMask); // set count to y - 1 where y != 0

    // build the mask for count > 0
    __pp_mask countIsPositiveMask;
    _pp_vgt_int(countIsPositiveMask, count, zero_i, expIsNotZeroMask); // 

    // loop until all lanes are done
    while (_pp_cntbits(countIsPositiveMask) > 0) {
      _pp_vmult_float(result, result, x, countIsPositiveMask);
      _pp_vsub_int(count, count, one_i, countIsPositiveMask);
      _pp_vgt_int(countIsPositiveMask, count, zero_i, expIsNotZeroMask); // update the mask
    }

    // clamp the result
    __pp_mask needClampMask;
    _pp_vgt_float(needClampMask, result, clamp_val, activeMask);       
    _pp_vmove_float(result, clamp_val, needClampMask);                 

    // store the result
    _pp_vstore_float(output + i, result, activeMask);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // Vectorized array reduction.
  //
  // Initialize a mask with all bits set (all lanes active)
  __pp_mask maskAll = _pp_init_ones();
  __pp_vec_float sum_vec = _pp_vset_float(0.f); 
  __pp_vec_float loaded_vals; 

  // Process the largest part of the array that is a multiple of VECTOR_WIDTH
  int limit = N - (N % VECTOR_WIDTH);
  for (int i = 0; i < limit; i += VECTOR_WIDTH)
  {
    _pp_vload_float(loaded_vals, values + i, maskAll);
    _pp_vadd_float(sum_vec, sum_vec, loaded_vals, maskAll);
  }

  // Sum the elements within the accumulator vector 
  for (int i = 0; i < log2(VECTOR_WIDTH); ++i)
  {
    _pp_hadd_float(sum_vec, sum_vec);
    _pp_interleave_float(sum_vec, sum_vec);
  }

  // After reduction, the total sum of the vectorized part is in the first element.
  float total_sum = sum_vec.value[0];

  // Add any remaining elements that didn't fit into a full vector.
  for (int i = limit; i < N; ++i)
  {
    total_sum += values[i];
  }

  return total_sum;

}
