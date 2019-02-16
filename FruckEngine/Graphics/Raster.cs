using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using FruckEngine.Helpers;
using OpenTK.Graphics.OpenGL;
using PixelFormat = System.Drawing.Imaging.PixelFormat;

namespace FruckEngine.Graphics {
    public class Raster {
        public int Width, Height;
        public byte[] Pixels;
        public Texture Texture { get; private set; }

        public Raster(int width, int height) {
            Width = width;
            Height = height;
            Pixels = new byte[Width * Height * 3];
        }

        public Raster(string path) {
            var bmp = new Bitmap(path);
            Width = bmp.Width;
            Height = bmp.Height;
            Pixels = new byte[Width * Height * 3];
            var data = bmp.LockBits(new Rectangle(0, 0, Width, Height), ImageLockMode.ReadOnly,
                PixelFormat.Format24bppRgb);
            Marshal.Copy(data.Scan0, Pixels, 0, Width * Height * 3);
            bmp.UnlockBits(data);
        }

        public void AttachTexture() {
            Texture = new Texture();
            Texture.SetFilters(TextureMinFilter.Nearest, TextureMagFilter.Nearest);
            //Texture.SetWrapping(TextureWrapMode.Repeat, TextureWrapMode.Repeat);
            Texture.InternalFormat = PixelInternalFormat.Rgb;
            Texture.Format = OpenTK.Graphics.OpenGL.PixelFormat.Bgr;
            Texture.Target = TextureTarget.Texture2D;
            TextureHelper.LoadDataIntoTexture(Texture, Width, Height, Pixels);
        }

        public void PushToTexture() {
            if (Texture == null) return;
            Texture.Bind();
            GL.TexImage2D(TextureTarget.Texture2D, 0, Texture.InternalFormat, Width, Height, 0, Texture.Format,
                PixelType.UnsignedByte, Pixels);
        }

        public void DrawText(string s, int x, int y, byte r, byte g, byte b) {
            FontHelper.Print(this, s, x, y, r, g, b);
        }

        public void Clear(int c) {
            for (int s = Width * Height, p = 0; p < s; p++) Pixels[p] = 0;
        }

        public void SetPixel(uint x, uint y, byte r, byte g, byte b) {
            uint offset = (uint) ((x + y * Width) * 3);
            Pixels[offset++] = r;
            Pixels[offset++] = g;
            Pixels[offset] = b;
        }
    }

    public static class FontHelper {
        private static Raster FontAtlas = null;

        private static string Characters =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_-+={}[];:<>,.?/\\ ";

        private static int[] FontRedir = null;

        private static void LoadCharacters() {
            FontAtlas = new Raster("Assets/Sprites/font.png");
            FontRedir = new int[256];

            for (int i = 0; i < 256; i++) FontRedir[i] = 0;
            for (int i = 0; i < Characters.Length; i++) {
                int l = Characters[i];
                FontRedir[l & 255] = i;
            }
        }

        public static void Print(Raster raster, string s, int x, int y, byte r, byte g, byte b) {
            if (FontAtlas == null) LoadCharacters();
            for (int i = 0; i < s.Length; i++) {
                int f = FontRedir[s[i] & 255];
                uint wx = (uint) (x + i * 12);
                uint wy = (uint) y;
                int src = f * 12;
                for (int v = 0; v < FontAtlas.Height; v++, src += FontAtlas.Width, wy += 1)
                for (int u = 0; u < 12; u++) {
                    if ((FontAtlas.Pixels[src + u] & 0xffffff) != 0) {
                        raster.SetPixel(wx, wy, r, g, b);
                    }
                }
            }
        }
    }
}