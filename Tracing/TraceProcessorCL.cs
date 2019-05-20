using Cloo;
using FruckEngine.Graphics;
using FruckEngine.Helpers;
using OpenTK.Graphics.OpenGL;
using Template;

namespace clrays {
    public enum TraceType
    {
        Real, AA
    }

    public class TraceProcessorCL {
        private readonly OpenCLProgram program;
        private TraceAaKernel traceAAKernel;
        private TraceKernel traceKernel;
        private VoidKernel<float> clearKernel;
        private ResultKernel<int> imageKernel;
        private Texture renderTexture;
        private readonly TraceType type;

        public TraceProcessorCL(uint width, uint height, uint AA, Scene scene, TraceType type) {
            this.type = type;
            program = new OpenCLProgram("Assets/Kernels/raytrace.cl");
            switch (type)
            {
                case TraceType.Real:
                    traceKernel = new TraceKernel("raytracing", program, scene, width, height);

                    break;
                case TraceType.AA:
                    traceAAKernel = new TraceAaKernel("raytracingAA", program, scene, width, height, AA);
                    clearKernel = new ClearKernel("clear", program, traceAAKernel.GetBuffer(), width, height);
                    imageKernel = new ImageKernel("image_from_floatmap", program, traceAAKernel.GetBuffer(), width, height, AA);
                    break;
            }
            //texture
            renderTexture = new Texture();
            renderTexture.Construct();
            renderTexture.Width = (int)width;
            renderTexture.Height = (int)height;
            renderTexture.InternalFormat = PixelInternalFormat.Rgba;
            renderTexture.Format = PixelFormat.Bgra;
            renderTexture.Target = TextureTarget.Texture2D;
            renderTexture.PixelType = PixelType.UnsignedByte;
            renderTexture.SetFilters(TextureMinFilter.Linear, TextureMagFilter.Linear);
            renderTexture.SetWrapping(TextureWrapMode.ClampToEdge, TextureWrapMode.ClampToEdge);
            Info.PrintInfo();
        }

        public void Render() {
            var events = new ComputeEventList();
            int[] image;
            switch (type)
            {
                case TraceType.Real:
                    traceKernel.Update();
                    traceKernel.Execute(events);
                    image = traceKernel.GetResult();
                    break;
                case TraceType.AA:
                    clearKernel.Execute(events);
                    traceAAKernel.Update();
                    traceAAKernel.Execute(events);
                    imageKernel.Execute(events);
                    image = imageKernel.GetResult();
                    break;
                default:
                    image = new int[] { };
                    break;
            }
            renderTexture.Bind();
            TextureHelper.LoadDataIntoTexture(renderTexture, renderTexture.Width, renderTexture.Height, image);
            renderTexture.Activate(0);
        }
    }
}
