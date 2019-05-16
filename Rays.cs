using System;
using System.Diagnostics;
using FruckEngine;
using FruckEngine.Graphics;
using OpenTK;

namespace clrays {
    public class Rays : Game {
        private Raster _raster;
        private Shader _shader;
        private TraceProcessorCL _processor = null;
        private int _generation;
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

            Scene scene = new Scene();
            scene.AddTexture("wood", "Assets/Textures/wood.png");
            scene.Add(new Plane
            {
                Pos = new Vector3(0, -1, 0),
                Nor = Vector3.UnitY,
                Mat = new Material
                {
                    Col = Vector3.One,
                    Reflectivity = 0.1f,
                    Shininess = 16f,
                }
            });
            scene.Add(new Sphere
            {
                Pos = new Vector3(1, 0, -5),
                Rad = 1f,
                Mat = new Material
                {
                    Col = new Vector3(0.9f, 0.1f, 0.1f).Normalized(),
                    Reflectivity = 0.0f,
                    Shininess = 512f,
                }
            });
            scene.Add(new Sphere
            {
                Pos = new Vector3(-1, 0, -5),
                Rad = 1f,
                Mat = new Material
                {
                    Col = new Vector3(0.1f, 0.1f, 0.9f).Normalized(),
                    Reflectivity = 1.0f,
                    Shininess = 2048f,
                }
            });
            scene.Add(new Light
            {
                Pos = new Vector3(0f,2f,-3),
                Intensity = 100,
                Col = Vector3.One,
            });
            _processor = new TraceProcessorCL((uint)Width, (uint)Height, 1, scene, TraceType.Real);
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
