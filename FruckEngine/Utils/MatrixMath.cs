using OpenTK;

namespace FruckEngine.Utils {
    public static class MatrixMath {
        public static Matrix3 ZoomAt2D(Matrix3 matrix, Vector2 point, Vector2 zoom) {
            return Translate2D(Translate2D(matrix, point) * Matrix3.CreateScale(zoom.X, zoom.Y, 1), -point);
        }

        public static Matrix3 Translate2D(Matrix3 matrix, Vector2 amount) {
            return matrix * new Matrix3(Vector3.UnitX, Vector3.UnitY, new Vector3(amount.X, amount.Y, 1));
        }
    }
}