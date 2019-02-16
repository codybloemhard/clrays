using FruckEngine.Helpers;
using FruckEngine.Structs;

namespace FruckEngine.Graphics {
    /// <summary>
    /// Primitives that are used as render object. For example plane for post effects
    /// </summary>
    public static class Projection {
        private static Mesh ProjectionPlane = null;

        public static void ProjectPlane() {
            if (ProjectionPlane == null) ProjectionPlane = DefaultModels.GetPlane(true);
            ProjectionPlane.Draw(null);
        }
    }
}