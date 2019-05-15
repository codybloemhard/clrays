using Cloo;
using FruckEngine.Graphics;
using FruckEngine.Helpers;
using OpenTK.Graphics.OpenGL;
using Template;

namespace clrays {
    public class TraceProcessorCL {
        private readonly OpenCLProgram program;
        private ResultKernel<float> traceAAKernel;
        private VoidKernel<float> clearKernel;
        private ResultKernel<int> imageKernel;
        public Texture renderTexture;

        public TraceProcessorCL(uint width, uint height, uint AA, Scene scene) {
            program = new OpenCLProgram("Assets/Kernels/raytrace.cl");
            traceAAKernel = new TraceAaKernel(program, scene, width, height, AA);
            clearKernel = new ClearKernel(program, traceAAKernel.GetBuffer(), width, height);
            imageKernel = new ImageKernel(program, traceAAKernel.GetBuffer(), width, height);
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
        }

        public void Render() {
            var events = new ComputeEventList();
            clearKernel.Execute(events);
            traceAAKernel.Execute(events);
            imageKernel.Execute(events);
            var image = imageKernel.GetResult();
            renderTexture.Bind();
            TextureHelper.LoadDataIntoTexture(renderTexture, renderTexture.Width, renderTexture.Height, image);
            renderTexture.Activate(0);
        }
    }
}
