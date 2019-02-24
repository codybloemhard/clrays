using System;
using System.Diagnostics;
using FruckEngine;
using FruckEngine.Graphics;
using FruckEngine.Utils;
using OpenTK;
using OpenTK.Input;
using Template;

namespace clrays {
    public class Rays : Game {
        private Raster _raster;
        private Shader _shader;
        private TraceProcessorCL _processor = null;
        private int _generation = 0;
        private Stopwatch _timer = new Stopwatch();

        public override void Init() {
            base.Init();
            _raster = new Raster(Width, Height);
            _raster.AttachTexture();

            _shader = Shader.Create("Assets/Shaders/rays_vs.glsl", "Assets/Shaders/rays_fs.glsl");
            _shader.Use();
            _shader.AddUniformVar("uTexture");
            _shader.AddUniformVar("mTransform");
            _shader.SetInt("uTexture", 0);

            string kernel = "Assets/Kernels/raytrace.cl";

            Scene scene = new Scene();
            scene.Add(new Plane(0,-1,0,0,1,0));
            scene.Add(new Sphere(1, 0, -5, 1));
            scene.Add(new Sphere(-1, 0, -5, 1));
            scene.Add(new Light(0, 2, -4, 2));

            _processor = new TraceProcessorCL(Width, Height, scene, kernel);
        }

        public override void Render(double dt)
        {
            _timer.Restart();
            _shader.Use();
            _processor.Render();
            Projection.ProjectPlane();
            long simulateTime = _timer.ElapsedMilliseconds;
            Console.WriteLine($"Generation: {_generation++}: Time: {simulateTime}ms");
        }

        public override void Resize(int width, int height) {
            base.Resize(width, height);
            _raster = new Raster(Width, Height);
        }

        public override void OnMouseScroll(double ds) {
            base.OnMouseScroll(ds);
        }

        public override void OnMouseButton(bool down) {
            base.OnMouseButton(down);
        }

        public override void OnMouseMove(double dx, double dy) {
            base.OnMouseMove(dx, dy);
        }
    }
}