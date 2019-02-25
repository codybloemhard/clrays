using System;
using System.Collections.Generic;
using OpenTK;
using System.Drawing;

namespace clrays
{
    public struct Material
    {
        public Vector3 Col;
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
                Mat.Col.X, Mat.Col.Y, Mat.Col.Z };
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
                Mat.Col.X, Mat.Col.Y, Mat.Col.Z };
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
            materialSize = 3,
            lightSize = 7,
            sphereSize = 4 + materialSize,
            planeSize = 6 + materialSize;

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

        public float[] GetBuffers()
        {
            int len = lights.Count * lightSize;
            len += spheres.Count * sphereSize;
            len += planes.Count * planeSize;
            var res = new float[len];
            int i = 0;
            Bufferize(res, ref i, lights, lightSize);
            Bufferize(res, ref i, spheres, sphereSize);
            Bufferize(res, ref i, planes, planeSize);
            return res;
        }

        public int[] GetParamsBuffer()
        {
            var res = new int[3 * 3];
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
