using FruckEngine.Graphics;

namespace FruckEngine.Helpers {
    /// <summary>
    /// A collection of standard shaders. All the uniforms are also registered
    /// </summary>
    public static class DefaultShaders {
        public static Shader CreateDefaultPlane() {
            var shader = Shader.Create("Assets/Shaders/plane.vs.glsl", "Assets/Shaders/gradient.fs.glsl");
            return shader;
        }
        
        public static Shader CreateDefaultTexture() {
            var shader = Shader.Create("Assets/Shaders/plane.vs.glsl", "Assets/Shaders/texture_default.fs.glsl");
            shader.AddUniformVar("uTexture");
            return shader;
        }
    }
}