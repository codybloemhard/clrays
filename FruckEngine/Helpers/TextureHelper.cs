using System.Runtime.InteropServices;
using FruckEngine.Graphics;
using OpenTK.Graphics.OpenGL;

namespace FruckEngine.Helpers {
    /// <summary>
    /// A helper for creatign and reusing textures
    /// </summary>
    public static class TextureHelper {
        /// <summary>
        /// Loads given data into the texture since a pointer to memory is needed which is easy to obtain in c++
        /// not so in c#
        /// </summary>
        /// <param name="texture"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="data"></param>
        /// <typeparam name="T"></typeparam>
        public static void LoadDataIntoTexture<T>(Texture texture, int width, int height, T[] data) {
            var gch = GCHandle.Alloc(data, GCHandleType.Pinned);
            try {
                var addr = gch.AddrOfPinnedObject();
                texture.Load(width, height, addr);
            } finally {
                gch.Free();
            }
        }
    }
}