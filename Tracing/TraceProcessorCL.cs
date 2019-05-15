using Cloo;
using FruckEngine.Graphics;
using FruckEngine.Helpers;
using OpenTK.Graphics.OpenGL;
using Template;

namespace clrays {
    public class TraceProcessorCL {
        private readonly OpenCLProgram program;

        private OpenCLKernel drawKernel;
        private readonly long[] drawKernelWork;
        private readonly float[] floatmap;
        private OpenCLBuffer<float> float_buffer;
        private readonly OpenCLBuffer<int> scene_params;
        private readonly OpenCLBuffer<float> scene_items;

        private OpenCLKernel clearKernel;
        private readonly long[] clearKernelWork;

        private OpenCLKernel imageKernel;
        private readonly long[] imageKernelWork;
        private readonly int[] imagemap;
        private OpenCLBuffer<int> image_buffer;
        public Texture renderTexture;

        public TraceProcessorCL(int width, int height, uint AA, Scene scene) {
            program = new OpenCLProgram("Assets/Kernels/raytrace.cl");
            floatmap = new float[width * height * 3];
            float_buffer = new OpenCLBuffer<float>(program, floatmap);
            //make kernel and set args
            drawKernel = new OpenCLKernel(program, "render");
            drawKernel.SetArgument(0, float_buffer);
            //make scene buffers
            var scene_raw = scene.GetBuffers();
            var scene_params_raw = scene.GetParamsBuffer();
            scene_params = new OpenCLBuffer<int>(program, scene_params_raw);
            scene_items = new OpenCLBuffer<float>(program, scene_raw);
            //set constants
            drawKernel.SetArgument(1, (uint)width);
            drawKernel.SetArgument(2, (uint)height);
            drawKernel.SetArgument(3, AA);
            //set arrays
            drawKernel.SetArgument(4, scene_params);
            drawKernel.SetArgument(5, scene_items);
            //work
            drawKernelWork = new long[] {width*AA, height*AA};
            // clear kernel
            clearKernel = new OpenCLKernel(program, "clear");
            clearKernel.SetArgument(0, float_buffer);
            clearKernel.SetArgument(1, (uint)width);
            clearKernel.SetArgument(2, (uint)height);
            clearKernelWork = new long[] { width, height };
            // image kernel
            imagemap = new int[width * height];
            image_buffer = new OpenCLBuffer<int>(program, imagemap);
            imageKernel = new OpenCLKernel(program, "image_from_floatmap");
            imageKernel.SetArgument(0, float_buffer);
            imageKernel.SetArgument(1, image_buffer);
            imageKernel.SetArgument(2, (uint)width);
            imageKernel.SetArgument(3, (uint)height);
            imageKernelWork = new long[] { width, height };
            //texture
            renderTexture = new Texture();
            renderTexture.Construct();
            renderTexture.Width = width;
            renderTexture.Height = height;
            renderTexture.InternalFormat = PixelInternalFormat.Rgba;
            renderTexture.Format = PixelFormat.Bgra;
            renderTexture.Target = TextureTarget.Texture2D;
            renderTexture.PixelType = PixelType.UnsignedByte;
            renderTexture.SetFilters(TextureMinFilter.Linear, TextureMagFilter.Linear);
            renderTexture.SetWrapping(TextureWrapMode.ClampToEdge, TextureWrapMode.ClampToEdge);
        }

        public void Render() {
            var events = new ComputeEventList();
            clearKernel.Execute(clearKernelWork, events);
            drawKernel.Execute(drawKernelWork, events);
            imageKernel.Execute(imageKernelWork, events);
            image_buffer.CopyFromDevice();
            renderTexture.Bind();
            TextureHelper.LoadDataIntoTexture(renderTexture, renderTexture.Width, renderTexture.Height, imagemap);
            renderTexture.Activate(0);
        }
    }
}
