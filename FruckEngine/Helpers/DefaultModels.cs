using System.Collections.Generic;
using FruckEngine.Graphics;
using FruckEngine.Structs;
using OpenTK;

namespace FruckEngine.Helpers {
    /// <summary>
    /// Default models / primitives
    /// </summary>
    /// Make a plane with a wider range of UV
    public static class DefaultModels {
        public static Mesh GetPlane(bool vertical) {
            List<Vertex> vertices = new List<Vertex> {
                new Vertex(new Vector3(-1, 1, 0), new Vector2(0.0f, 0.0f)),
                new Vertex(new Vector3(-1, -1, 0), new Vector2(0.0f, 1.0f)),
                new Vertex(new Vector3(1, 1, 0), new Vector2(1.0f, 0.0f)),
                new Vertex(new Vector3(1, -1, 0), new Vector2(1.0f, 1.0f))
            };

            var indices = new List<uint>() {0, 1, 2, 1, 2, 3};
            return new Mesh(vertices.ToArray(), indices.ToArray());
        }
    }
}