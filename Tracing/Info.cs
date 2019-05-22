using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace clrays
{
    public static class Info
    {
        public static List<(string, uint)> Textures = new List<(string, uint)>();
        public static uint MetaSize, SceneSize;
        public static uint IntMapSize, FloatMapSize;
        private static Stopwatch watch = new Stopwatch();
        private static List<(string, int)> times = new List<(string, int)>();

        public static void PrintInfo()
        {
            Console.WriteLine($"Metadata: {MetaSize} B.");
            Console.WriteLine($"Scene:    {SceneSize} B.");
            Console.WriteLine($"Int FB:   {IntMapSize} B.");
            Console.WriteLine($"Float FB: {FloatMapSize} B.");
            uint sum = 0;
            for(int i = 0; i < Textures.Count; i++)
            {
                var name = Textures[i].Item1;
                var size = Textures[i].Item2;
                sum += size;
                Console.WriteLine($"Texture{i} : {name} : {size} B.");
            }
            Console.WriteLine("Totalsize: ");
            PrintSizeVerbose(sum);
            Console.WriteLine("Grand Total: ");
            sum += MetaSize + SceneSize + IntMapSize + FloatMapSize;
            PrintSizeVerbose(sum);
            int last = 0;
            for(int i = 0; i < times.Count; i++)
            {
                int elapsed = times[i].Item2 - last;
                last = times[i].Item2;
                Console.WriteLine($"{times[i].Item1}: {elapsed} ms.");
            }
            Console.WriteLine($"Total: {last}");
            watch.Stop();//why not
        }

        private static void PrintSizeVerbose(uint size)
        {
            Console.WriteLine($"    {size / Math.Pow(1024, 0)} B.");
            Console.WriteLine($"    {size / Math.Pow(1024, 1)} KB.");
            Console.WriteLine($"    {size / Math.Pow(1024, 2)} MB.");
            Console.WriteLine($"    {size / Math.Pow(1024, 3)} GB.");
        }

        public static void StartTime()
        {
            watch.Start();
        }

        public static void SetTimePoint(string name)
        {
            times.Add((name,(int)watch.ElapsedMilliseconds));
        }
    }
}
