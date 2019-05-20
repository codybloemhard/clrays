using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace clrays
{
    public enum TexType
    {
        Vector3c8bpc,
        Scalar8b,
    }

    public struct TraceTex
    {
        public byte[] Pixels;
        public int Width, Height;

        public static TraceTex VectorTex(string path)
        {
            TraceTex tex = new TraceTex();
            var bmp = new Bitmap(path);
            tex.Width = bmp.Width;
            tex.Height = bmp.Height;
            tex.Pixels = new byte[tex.Width * tex.Height * 3];
            var data = bmp.LockBits(new Rectangle(0, 0, tex.Width, tex.Height), ImageLockMode.ReadOnly,
                PixelFormat.Format24bppRgb);
            Marshal.Copy(data.Scan0, tex.Pixels, 0, tex.Width * tex.Height * 3);
            bmp.UnlockBits(data);
            return tex;
        }

        public static TraceTex ScalarTex(string path)
        {
            TraceTex tex = new TraceTex();
            var bmp = new Bitmap(path);
            tex.Width = bmp.Width;
            tex.Height = bmp.Height;
            tex.Pixels = new byte[tex.Width * tex.Height];
            var temp = new byte[tex.Width * tex.Height * 3];
            var data = bmp.LockBits(new Rectangle(0, 0, tex.Width, tex.Height), ImageLockMode.ReadOnly,
                PixelFormat.Format24bppRgb);
            Marshal.Copy(data.Scan0, temp, 0, tex.Width * tex.Height * 3);
            bmp.UnlockBits(data);
            for(int i = 0; i < tex.Pixels.Length; i++)
            {
                long val = 0;
                val += temp[i * 3 + 0];
                val += temp[i * 3 + 1];
                val += temp[i * 3 + 2];
                val /= 3;
                tex.Pixels[i] = (byte)val;
            }
            return tex;
        }
    }
}
