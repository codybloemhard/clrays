using System;
using System.Runtime.InteropServices;
using FruckEngine.Structs;
using FruckEngine.Utils;
using OpenTK.Graphics.OpenGL;

namespace FruckEngine.Graphics {
    /// <summary>
    /// Abstraction for a mesh. Our meshes are not much more than some 3d model data and its material.
    /// Meshes do not allow any other elements than triangle. SO make sure to convert any quads beforehand.
    /// </summary>
    public class Mesh {
        private int VAO, VBO, EBO;

        public Vertex[] Vertices;
        public uint[] Indices;

        /// <summary>
        /// Create mesh with all the minimum properties to render it. Otherwise no point in making one
        /// </summary>
        /// <param name="vertices"></param>
        /// <param name="indices"></param>
        public Mesh(Vertex[] vertices, uint[] indices) {
            Vertices = vertices;
            Indices = indices;
            Init();
        }

        /// <summary>
        /// Inits the mesh and uploads all vertices to gpu
        /// </summary>
        protected virtual void Init() {
            VAO = GL.GenVertexArray();
            VBO = GL.GenBuffer();
            EBO = GL.GenBuffer();

            GL.BindVertexArray(VAO);

            // Upload vertices array object
            GL.BindBuffer(BufferTarget.ArrayBuffer, VBO);
            GL.BufferData(BufferTarget.ArrayBuffer, (IntPtr) (Vertices.Length * Mem.SizeOf(typeof(Vertex))), Vertices,
                BufferUsageHint.StaticDraw);

            // Upload indices
            GL.BindBuffer(BufferTarget.ElementArrayBuffer, EBO);
            GL.BufferData(BufferTarget.ElementArrayBuffer, Indices.Length * Mem.SizeOf(typeof(uint)), Indices,
                BufferUsageHint.StaticDraw);

            // vertex positions
            GL.EnableVertexAttribArray(0);
            GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, Mem.SizeOf(typeof(Vertex)),
                Marshal.OffsetOf(typeof(Vertex), "Position"));
            // vertex texture coords
            GL.EnableVertexAttribArray(1);
            GL.VertexAttribPointer(1, 2, VertexAttribPointerType.Float, false, Mem.SizeOf(typeof(Vertex)),
                Marshal.OffsetOf(typeof(Vertex), "UV"));

            GL.BindVertexArray(0);
        }

        /// <summary>
        /// Draws the mesh if shading mode matches the material or is any.
        /// Also material will be bound of shading is enabled
        /// </summary>
        /// <param name="shader"></param>
        public virtual void Draw(Shader shader) {
            // Draw
            GL.BindVertexArray(VAO);
            GL.DrawElements(BeginMode.Triangles, Indices.Length, DrawElementsType.UnsignedInt, 0);
            GL.BindVertexArray(0);
        }

        /// <summary>
        /// Deletes the mesh. Make sure that there are no clones in the world first.
        /// </summary>
        public virtual void Destroy() {
            GL.DeleteBuffer(VBO);
            GL.DeleteBuffer(EBO);
            GL.DeleteVertexArray(VAO);
        }
    }
}