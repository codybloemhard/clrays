using System.Collections.Generic;
using OpenTK;
using FruckEngine.Graphics;

namespace clrays
{
    public class Material
    {
        public Vector3 Col;
        public float
            Reflectivity,
            Shininess;
        public int Texture;
        private float texScale;
        public float TexScale {
            get { return texScale; }
            set { texScale = 1f / value; }
        }

        public Material()
        {
            Col = Vector3.One;
            texScale = 1f;
        }
    }

    public interface SceneItem
    {
        Vector3 Pos { get; set; }
        float[] GetData();
    }

    public struct Plane : SceneItem
    {
        public Vector3 Pos { get; set; }
        public Vector3 Nor { get; set; }
        public Material Mat { get; set; }

        public float[] GetData()
        {
            return new float[] { Pos.X, Pos.Y, Pos.Z, 
                Nor.X, Nor.Y, Nor.Z,
                Mat.Col.X, Mat.Col.Y, Mat.Col.Z, Mat.Reflectivity, 
                Mat.Shininess, Mat.Texture, Mat.TexScale };
        }
    }

    public struct Sphere : SceneItem
    {
        public Vector3 Pos { get; set; }
        public float Rad { get; set; }
        public Material Mat { get; set; }

        public float[] GetData()
        {
            return new float[] { Pos.X, Pos.Y, Pos.Z, Rad,
                Mat.Col.X, Mat.Col.Y, Mat.Col.Z, Mat.Reflectivity,
                Mat.Shininess, Mat.Texture, Mat.TexScale };
        }
    }

    public struct Box : SceneItem
    {
        public Vector3 Pos { get; set; }
        public Vector3 Size { get; set; }
        public Material Mat { get; set; }

        public float[] GetData()
        {
            var hs = Size / 2f;
            return new float[] {
                Pos.X - hs.X, Pos.Y - hs.Y, Pos.Z - hs.Z,
                Pos.X + hs.X, Pos.Y + hs.Y, Pos.Z + hs.Z,
                Mat.Col.X, Mat.Col.Y, Mat.Col.Z, Mat.Reflectivity,
                Mat.Shininess, Mat.Texture, Mat.TexScale };
        }
    }

    public struct Light : SceneItem
    {
        public Vector3 Pos { get; set; }
        public float Intensity { get; set; }
        public Vector3 Col { get; set; }

        public float[] GetData()
        {
            return new float[] { Pos.X, Pos.Y, Pos.Z, Intensity,
                Col.X, Col.Y, Col.Z };
        }
    }

    public class Scene
    {
        public const int
            sceneSize = 4,
            materialSize = 7,
            lightSize = 7,
            sphereSize = 4 + materialSize,
            planeSize = 6 + materialSize,
            boxSize = 6 + materialSize;

        private List<SceneItem> 
            spheres,
            lights,
            planes,
            boxes;
        private int nextTexture;
        private Dictionary<string, int> texturesIds;
        private List<Raster> textures;
        private int skybox;
        public Vector3 SkyCol { get; set; }

        public Scene()
        {
            spheres = new List<SceneItem>();
            lights = new List<SceneItem>();
            planes = new List<SceneItem>();
            boxes = new List<SceneItem>();
            nextTexture = 0;
            texturesIds = new Dictionary<string, int>();
            textures = new List<Raster>();
        }

        public float[] GetBuffers()
        {
            int len = lights.Count * lightSize;
            len += spheres.Count * sphereSize;
            len += planes.Count * planeSize;
            len += boxes.Count * boxSize;
            var res = new float[len];
            int i = 0;
            Bufferize(res, ref i, lights, lightSize);
            Bufferize(res, ref i, spheres, sphereSize);
            Bufferize(res, ref i, planes, planeSize);
            Bufferize(res, ref i, boxes, boxSize);
            return res;
        }

        public int[] GetParamsBuffer()
        {
            var res = new int[4 * 3 + sceneSize];
            int i = 0;
            res[0] = lightSize;
            res[1] = lights.Count;
            res[2] = i; i += lights.Count * lightSize;
            res[3] = sphereSize;
            res[4] = spheres.Count;
            res[5] = i; i += spheres.Count * sphereSize;
            res[6] = planeSize;
            res[7] = planes.Count;
            res[8] = i; i += planes.Count * planeSize;
            res[9] = boxSize;
            res[10] = boxes.Count;
            res[11] = i; i += boxes.Count * boxSize;
            //scene
            res[12] = skybox;
            res[13] = SkyCol.X.GetHashCode();
            res[14] = SkyCol.Y.GetHashCode();
            res[15] = SkyCol.Z.GetHashCode();
            System.Console.WriteLine($"skybox: {skybox}");
            return res;
        }

        public byte[] GetTexturesBuffer()
        {
            uint size = 0;
            for (int i = 0; i < textures.Count; i++)
                size += (uint)textures[i].Pixels.Length;
            var res = new byte[size];
            int start = 0;
            for (int i = 0; i < textures.Count; i++)
            {
                int len = textures[i].Pixels.Length;
                for (int j = 0; j < len; j++)
                    res[start + j] = textures[i].Pixels[j];
                start += len;
            }
            return res;
        }

        public int[] GetTextureParamsBuffer()
        {
            var res = new int[textures.Count * 3];
            int start = 0;
            for(int i = 0; i < textures.Count; i++)
            {
                res[i * 3 + 0] = start;
                res[i * 3 + 1] = textures[i].Width;
                res[i * 3 + 2] = textures[i].Height;
                start += textures[i].Pixels.Length;
            }
            return res;
        }

        private void Bufferize(float[] arr, ref int start, List<SceneItem> list, int stride)
        {
            for(int i = 0; i < list.Count; i++)
            {
                int off = i * stride;
                var data = list[i].GetData();
                for (int j = 0; j < stride; j++)
                    arr[start + off + j] = data[j];
            }
            start += list.Count * stride;
        }

        public void AddTexture(string name, string path)
        {
            var raster = new Raster(path);
            texturesIds.Add(name, nextTexture++);
            textures.Add(raster);
        }

        public void Add(Sphere s)
        {
            spheres.Add(s);
        }

        public void Add(Plane p)
        {
            planes.Add(p);
        }

        public void Add(Light l)
        {
            lights.Add(l);
        }

        public void Add(Box b)
        {
            boxes.Add(b);
        }

        public int GetTexture(string name)
        {
            if (texturesIds.ContainsKey(name))
                return texturesIds[name] + 1;
            return 0;
        }

        public void SetSkybox(string name)
        {
            if (texturesIds.ContainsKey(name))
                skybox = texturesIds[name] + 1;
        }
    }
}
