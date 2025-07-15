pub extern crate nalgebra;

use nalgebra::{
    allocator::Allocator,
    constraint::{DimEq, ShapeConstraint},
    DefaultAllocator, Dim, Dyn, Matrix, Matrix2, OMatrix, Point2, RawStorageMut, RealField,
    Rotation2, SimilarityMatrix2, Translation2, SVD, U2,
};

#[inline]
pub fn from_point_slices<T: RealField>(
    from: &[Point2<T>],
    to: &[Point2<T>],
    eps: T,
    max_niter: usize,
) -> Option<SimilarityMatrix2<T>>
where
    DefaultAllocator: Allocator<U2, Dyn>,
{
    from_matrices(
        point_slice_to_matrix(from),
        point_slice_to_matrix(to),
        eps,
        max_niter,
    )
}

#[inline]
pub fn point_slice_to_matrix<T: RealField>(slice: &[Point2<T>]) -> OMatrix<T, U2, Dyn>
where
    DefaultAllocator: Allocator<U2, Dyn>,
{
    OMatrix::<T, U2, Dyn>::from_iterator(
        slice.len(),
        slice
            .iter()
            .map(|p| p.coords.iter())
            .flatten()
            .map(|v| v.to_owned()),
    )
}

#[inline]
pub fn from_matrices<T, D1, D2, S>(
    mut from: Matrix<T, U2, D1, S>,
    mut to: Matrix<T, U2, D2, S>,
    eps: T,
    max_niter: usize,
) -> Option<SimilarityMatrix2<T>>
where
    T: RealField,
    D1: Dim,
    D2: Dim,
    ShapeConstraint: DimEq<D1, D2>,
    S: RawStorageMut<T, U2, D1> + RawStorageMut<T, U2, D2>,
{
    let size_recip = T::from_usize(from.column_iter().count())
        .expect("Cannot convert usize to T")
        .recip();

    let mean_from = from.column_mean();
    let mean_to = to.column_mean();

    for mut col in from.column_iter_mut() {
        col -= &mean_from;
    }
    for mut col in to.column_iter_mut() {
        col -= &mean_to;
    }

    let mut sigma = from
        .column_iter()
        .fold(T::zero(), |sigma, col| sigma + col.norm_squared());

    let mut cov = from
        .column_iter()
        .zip(to.column_iter())
        .fold(Matrix2::zeros(), |cov, (from_col, to_col)| {
            cov + (to_col * from_col.transpose())
        });

    sigma *= size_recip.to_owned();
    cov.scale_mut(size_recip);

    let det = cov.determinant();
    let SVD {
        u,
        v_t,
        singular_values,
    } = cov.try_svd_unordered(true, true, eps, max_niter)?;

    let u = u?;
    let v_t = v_t?;
    let d = Matrix2::from_diagonal(&singular_values);

    let mut s = Matrix2::identity();

    if det < T::zero() || (det == T::zero() && (u.determinant() * v_t.determinant()) < T::zero()) {
        let index = if d[(1, 1)] < d[(0, 0)] {
            (1, 1)
        } else {
            (0, 0)
        };

        s[index] = T::one().neg();
    }

    let scaling = if sigma != T::zero() {
        sigma.recip() * (d * &s).trace()
    } else {
        T::one()
    };

    let rotation = Rotation2::from_matrix_unchecked(u * s * v_t);
    let translation: Translation2<T> =
        (mean_to - (&rotation * mean_from).scale(scaling.to_owned())).into();

    Some(SimilarityMatrix2::from_parts(
        translation,
        rotation,
        scaling,
    ))
}
