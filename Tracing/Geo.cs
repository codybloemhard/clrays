using System;
using System.Collections.Generic;

namespace clrays
{
    public interface SceneItem
    {
        float[] data { get; set; }
    }

    public struct Plane : SceneItem
    {
        public float[] data { get; set; }

        public Plane(float x, float y, float z, float nx, float ny, float nz)
        {
            data = new float[] { x, y, z, nx, ny, nz };
        }
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
        public const int
            sphereSize = 4,
            lightSize = 4,
            planeSize = 6;

        private List<SceneItem> 
            spheres,
            lights,
            planes;

        public Scene()
        {
            spheres = new List<SceneItem>();
            lights = new List<SceneItem>();
            planes = new List<SceneItem>();
        }

        public float[][] GetBuffers()
        {
            var res = new float[3][];
            res[0] = Bufferize(spheres, sphereSize);
            res[1] = Bufferize(lights, lightSize);
            res[2] = Bufferize(planes, planeSize);
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

        public void Add(Plane p)
        {
            planes.Add(p);
        }

        public void Add(Light l)
        {
            lights.Add(l);
        }
    }
}
