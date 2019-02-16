using System.Runtime.InteropServices;
using OpenTK;

namespace FruckEngine.Structs
{
    /// <summary>
    /// A vertex with a few precalculated properties like tangent and bitangent
    /// </summary>
    [StructLayout(LayoutKind.Sequential)] public struct Vertex
    {
        public Vector3 Position;
        public Vector2 UV;

        public Vertex(Vector3 position, Vector2 uv) {
            Position = position;
            UV = uv;
        }
    }
}