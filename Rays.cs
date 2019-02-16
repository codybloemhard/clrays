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
        private GameProcessorCL _processor = null;
        private int _generation = 0;
        private Stopwatch _timer = new Stopwatch();

        public override void Init() {
            base.Init();
            _raster = new Raster(Width, Height);
            _raster.AttachTexture();

            _shader = Shader.Create("Assets/Shaders/game_of_life.vs.glsl", "Assets/Shaders/game_of_life.fs.glsl");
            _shader.Use();
            _shader.AddUniformVar("uTexture");
            _shader.AddUniformVar("mTransform");
            _shader.SetInt("uTexture", 0);

            _processor = new GameProcessorCL(Width, Height);
        }

        public override void Render(double dt)
        {
            _timer.Restart();
            _shader.Use();
            _processor.Render();
            Projection.ProjectPlane();
            long simulateTime = _timer.ElapsedMilliseconds;
            Console.WriteLine($"Generation: {_generation++}: Simulate: {simulateTime}ms, Render: {_timer.ElapsedMilliseconds - simulateTime}ms");
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