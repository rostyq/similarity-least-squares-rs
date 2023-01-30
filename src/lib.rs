pub extern crate nalgebra;

use nalgebra::{
    Matrix2, Point2, RealField, Rotation2, SMatrix, SimilarityMatrix2, Translation2, SVD,
};

pub fn from_point_arrays<T: RealField, const D: usize>(
    from: &[Point2<T>; D],
    to: &[Point2<T>; D],
) -> Result<SimilarityMatrix2<T>, &'static str> {
    from_smatrices(point_array_to_smatrix(from), point_array_to_smatrix(to))
}

pub fn point_array_to_smatrix<T: RealField, const D: usize>(
    slice: &[Point2<T>; D],
) -> SMatrix<T, 2, D> {
    SMatrix::<T, 2, D>::from_iterator(
        slice
            .iter()
            .map(|p| p.coords.iter())
            .flatten()
            .map(|v| v.to_owned()),
    )
}

pub fn from_smatrices<T: RealField, const D: usize>(
    mut from: SMatrix<T, 2, D>,
    mut to: SMatrix<T, 2, D>,
) -> Result<SimilarityMatrix2<T>, &'static str> {
    let size_recip = T::from_usize(D).ok_or("Cannot convert usize to T")?.recip();

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
    } = cov.svd_unordered(true, true);

    let u = u.ok_or("Cannot obtain the left-singular vectors of the SVD")?;
    let v_t = v_t.ok_or("Cannot obtain the right-singular vectors of the SVD")?;
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

    Ok(SimilarityMatrix2::from_parts(
        translation,
        rotation,
        scaling,
    ))
}
