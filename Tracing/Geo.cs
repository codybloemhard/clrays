using System;
using System.Collections.Generic;

namespace clrays
{
    public interface SceneItem
    {
        float[] data { get; set; }
    }

    public struct Sphere : SceneItem
    {
        public float[] data { get; set; }

        public Sphere(float x, float y, float z, float r)
        {
            data = new float[] { x, y, z, r};
        }
    }

    public struct Light : SceneItem
    {
        public float[] data { get; set; }

        public Light(float x, float y, float z, float p)
        {
            data = new float[] { x, y, z, p };
        }
    }

    public class Scene
    {
        public const int sphereSize = 4;
        public const int lightSize = 4;
        private List<SceneItem> spheres;
        private List<SceneItem> lights;

        public Scene()
        {
            spheres = new List<SceneItem>();
            lights = new List<SceneItem>();
        }

        public float[][] GetBuffers()
        {
            var res = new float[2][];
            res[0] = Bufferize(spheres, sphereSize);
            res[1] = Bufferize(lights, lightSize);
            return res;
        }

        private float[] Bufferize(List<SceneItem> list, int itemSize)
        {
            float[] res = new float[list.Count * itemSize];
            for(int i = 0; i < list.Count; i++)
            {
                int off = i * itemSize;
                var data = list[i].data;
                for (int j = 0; j < itemSize; j++)
                    res[off + j] = data[j];
            }
            return res;
        }

        public void Add(Sphere s)
        {
            spheres.Add(s);
        }

        public void Add(Light l)
        {
            lights.Add(l);
        }
    }
}
