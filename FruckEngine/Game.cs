using System.Drawing;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;

namespace FruckEngine {
    public abstract class Game {
        public double Time;
        public Vector2 MousePosition = Vector2.Zero;

        public int Width { get; private set; }
        public int Height { get; private set; }

        public virtual void Init() {
        }

        /// <summary>
        /// Update function. ALl the input and scheduled movement should happen here
        /// </summary>
        /// <param name="dt"></param>
        public virtual void Update(double dt) {
            Time += dt;
        }

        /// <summary>
        /// Render function. All the drawing should happen here.
        /// </summary>
        public abstract void Render(double dt);

        /// <summary>
        /// Resize everything 
        /// </summary>
        /// <param name="width"></param>
        /// <param name="height"></param>
        public virtual void Resize(int width, int height) {
            Width = width;
            Height = height;
        }

        /// <summary>
        /// Deletion of buffers etc should happen here
        /// </summary>
        public virtual void Destroy() {
        }

        /// <summary>
        /// Clear function called before the render call. 
        /// </summary>
        public virtual void Clear() {
            GL.ClearColor(Color.Black);
            GL.Enable(EnableCap.Texture2D);
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadIdentity();
            GL.MatrixMode(MatrixMode.Projection);
            GL.LoadIdentity();
        }


        public virtual void OnMouseMove(double dx, double dy) {
        }

        public virtual void OnMouseButton(bool down) {
        }

        public virtual void OnMouseScroll(double ds) {
        }

        public virtual void OnKeyboardUpdate(KeyboardState state) {
        }
    }
}