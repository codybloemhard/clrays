using System;
using System.Collections.Generic;

namespace clrays
{
    public struct Sphere
    {
        public float x, y, z, r;

        public Sphere(float x, float y, float z, float r)
        {
            this.x = x;
            this.y = y;
            this.z = z;
            this.r = r;
        }
    }

    public class Scene
    {
        public const int sphereSize = 4;
        public List<Sphere> spheres;

        public Scene()
        {
            spheres = new List<Sphere>();
        }

        public float[][] GetBuffers()
        {
            var res = new float[1][];
            res[0] = new float[spheres.Count * sphereSize];
            int j = 0;
            for(int i = 0; i < spheres.Count; i++)
            {
                res[0][j + 0] = spheres[i].x;
                res[0][j + 1] = spheres[i].y;
                res[0][j + 2] = spheres[i].z;
                res[0][j + 3] = spheres[i].r;
                j += sphereSize;
            }
            return res;
        }

        public void Add(Sphere s)
        {
            spheres.Add(s);
        }
    }
}
