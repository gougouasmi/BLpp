#ifndef DENSE_LINALG_ARRAY_HPP
#define DENSE_LINALG_ARRAY_HPP

namespace ArrayLinalg {

template <typename T, std::size_t xdim>
void DenseMatrixMultiply(const array<T, xdim * xdim> &matrix_data_rm,
                         const array<T, xdim> &input_vector,
                         array<T, xdim> &output) {
  int offset = 0;
  for (int i = 0; i < xdim; i++) {
    output[i] = 0;
    for (int j = 0; j < xdim; j++) {
      output[i] += matrix_data_rm[offset + j] * input_vector[j];
    }
    offset += xdim;
  }
}

} // namespace ArrayLinalg

#endif