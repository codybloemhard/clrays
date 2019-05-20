using System;
using System.Diagnostics;
using FruckEngine;
using FruckEngine.Graphics;
using OpenTK;
using OpenTK.Input;

namespace clrays {
    public class Rays : Game {
        private Raster _raster;
        private Shader _shader;
        private TraceProcessorCL _processor = null;
        private int _generation;
        private Stopwatch _timer = new Stopwatch();

        private Scene scene;
        private KeyboardState kstate;
        private readonly float speed = 1f;
        private bool canMove, wasDown;
        private Vector3 hor, ver;

        public override void Init() {
            base.Init();
            _raster = new Raster(Width, Height);
            _raster.AttachTexture();

            _shader = Shader.Create("Assets/Shaders/rays_vs.glsl", "Assets/Shaders/rays_fs.glsl");
            _shader.Use();
            _shader.AddUniformVar("uTexture");
            _shader.AddUniformVar("mTransform");
            _shader.SetInt("uTexture", 0);
            scene = new Scene();
            scene.AddTexture("wood", "Assets/Textures/wood.png");
            scene.AddTexture("sphere", "Assets/Textures/spheremap.jpg");
            scene.AddTexture("stone-alb", "Assets/Textures/stone-albedo.tif");
            scene.AddTexture("stone-nor", "Assets/Textures/stone-normal.tif");
            scene.AddTexture("stone-rou", "Assets/Textures/stone-rough.tif", TexType.Scalar8b);
            scene.AddTexture("tiles-alb", "Assets/Textures/tiles-albedo.tif");
            scene.AddTexture("tiles-nor", "Assets/Textures/tiles-normal.tif");
            scene.AddTexture("tiles-rou", "Assets/Textures/tiles-rough.tif", TexType.Scalar8b);
            scene.AddTexture("scifi-alb", "Assets/Textures/scifi-albedo.tif");
            scene.AddTexture("scifi-nor", "Assets/Textures/scifi-normal.tif");
            scene.AddTexture("scifi-rou", "Assets/Textures/scifi-rough.tif", TexType.Scalar8b);
            scene.AddTexture("scifi-met", "Assets/Textures/scifi-metal.tif", TexType.Scalar8b);
            scene.AddTexture("solar-alb", "Assets/Textures/solar-albedo.tif");
            scene.AddTexture("solar-nor", "Assets/Textures/solar-normal.tif");
            scene.AddTexture("solar-rou", "Assets/Textures/solar-rough.tif", TexType.Scalar8b);
            scene.AddTexture("solar-met", "Assets/Textures/solar-metal.tif", TexType.Scalar8b);
            scene.AddTexture("sky", "Assets/Textures/sky1.jpg");
            scene.SetSkybox("sky");
            scene.SkyCol = new Vector3(0.2f, 0.2f, 0.9f).Normalized();
            scene.SkyIntensity = 0.0f;
            scene.CamPos = Vector3.Zero;
            scene.CamDir = new Vector3(0f,0f,-1f).Normalized();
            scene.Add(new Plane
            {
                Pos = new Vector3(0, -1, 0),
                Nor = Vector3.UnitY,
                Mat = new Material
                {
                    Reflectivity = 0f,
                    Texture = scene.GetTexture("stone-alb"),
                    NormalMap = scene.GetTexture("stone-nor"),
                    RoughnessMap = scene.GetTexture("stone-rou"),
                    TexScale = 4f,
                }
            });
            scene.Add(new Sphere
            {
                Pos = new Vector3(2, 0, -5),
                Rad = 1f,
                Mat = new Material
                {
                    Reflectivity = 0f,
                    //Texture = scene.GetTexture("sphere"),
                    Texture = scene.GetTexture("tiles-alb"),
                    NormalMap = scene.GetTexture("tiles-nor"),
                    RoughnessMap = scene.GetTexture("tiles-rou"),
                }
            });
            scene.Add(new Sphere
            {
                Pos = new Vector3(0, 0, -5),
                Rad = 1f,
                Mat = new Material
                {
                    Reflectivity = 0.3f,
                    Texture = scene.GetTexture("solar-alb"),
                    NormalMap = scene.GetTexture("solar-nor"),
                    RoughnessMap = scene.GetTexture("solar-rou"),
                    MetalicMap = scene.GetTexture("solar-met"),
                }
            });
            scene.Add(new Sphere
            {
                Pos = new Vector3(-2, 0, -5),
                Rad = 1f,
                Mat = new Material
                {
                    //Reflectivity = 0f,
                    Texture = scene.GetTexture("scifi-alb"),
                    NormalMap = scene.GetTexture("scifi-nor"),
                    RoughnessMap = scene.GetTexture("scifi-rou"),
                    MetalicMap = scene.GetTexture("scifi-met"),
                }
            });
            scene.Add(new Light
            {
                Pos = new Vector3(0f,2f,-3),
                Intensity = 100,
                Col = Vector3.One,
            });
            _processor = new TraceProcessorCL((uint)Width, (uint)Height, 2, scene, TraceType.Real);
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

        public override void Update(double dt)
        {
            if (!canMove) return;
            float fdt = (float)dt;
            base.OnKeyboardUpdate(kstate);
            if (kstate.IsKeyDown(Key.A))
                scene.CamPos -= hor * speed * fdt;
            if (kstate.IsKeyDown(Key.D))
                scene.CamPos += hor * speed * fdt;
            if (kstate.IsKeyDown(Key.W))
                scene.CamPos += scene.CamDir * speed * fdt;
            if (kstate.IsKeyDown(Key.S))
                scene.CamPos -= scene.CamDir * speed * fdt;
            if (kstate.IsKeyDown(Key.E))
                scene.CamPos -= Vector3.UnitY * speed * fdt;
            if (kstate.IsKeyDown(Key.Q))
                scene.CamPos += Vector3.UnitY * speed * fdt;
            scene.Update();
        }

        public override void OnKeyboardUpdate(KeyboardState state)
        {
            kstate = state;
            wasDown |= state.IsKeyDown(Key.Space);
            if (state.IsKeyUp(Key.Space) && wasDown)
            {
                canMove = !canMove;
                wasDown = false;
            }
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

        public override void OnMouseMove(double _, double __) {
            var mid = new Vector2(Width / 2f, Height / 2f);
            float dx = mid.X - Mouse.GetCursorState().X;
            float dy = mid.Y - Mouse.GetCursorState().Y;
            Mouse.SetPosition(mid.X, mid.Y);
            if (!canMove) return;                                  
            var dir = scene.CamDir;
            hor = Vector3.Cross(dir, Vector3.UnitY).Normalized();
            ver = Vector3.Cross(hor, dir).Normalized();
            dir -= hor * (float)dx / Width * 2;
            dir += ver * (float)dy / Width * 2;
            scene.CamDir = dir.Normalized(); 
        }
    }
}
