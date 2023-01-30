#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::{Point2, SimilarityMatrix2, Vector2};

    use similarity_least_squares;

    #[test]
    fn check_find_similarity() {
        let from_points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(-2.0, 0.0),
            Point2::new(2.0, -0.5),
            Point2::new(-1.0, -1.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
        ];

        let angles: Vec<f32> = vec![
            0f32,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::FRAC_PI_3,
            std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_6,
        ];
        let scales: Vec<f32> = vec![1.0, 3f32.recip(), 0.5, 1.5, 2.0];
        let xs: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0];
        let ys: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0];

        for scale in scales.iter() {
            for angle in angles.iter() {
                for x in xs.iter() {
                    for y in ys.iter() {
                        let test = SimilarityMatrix2::new(Vector2::new(*x, *y), *angle, *scale);

                        let to_points: Vec<Point2<f32>> = from_points
                            .iter()
                            .map(|point| test.transform_point(point))
                            .collect();
                        
                        println!("{:?}", from_points);
                        println!("{:?}\n", to_points);

                        let transform = similarity_least_squares::from_point_arrays::<f32, 7>(
                            &from_points.clone().try_into().unwrap(),
                            &to_points.try_into().unwrap(),
                        ).unwrap();

                        assert_abs_diff_eq!(transform, test, epsilon = 0.001);
                    }
                }
            }
        }
    }
}
