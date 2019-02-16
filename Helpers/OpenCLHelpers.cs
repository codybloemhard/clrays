using System;
using System.Collections.Generic;
using System.IO;
using Cloo;

namespace clrays.Helpers {
    public static class OpenCLHelpers {
        /// <summary>
        /// This function attempts to find the best platform / device for OpenCL code execution.
        /// The best device is typically not the CPU, nor an integrated GPU. If no GPU is found,
        /// the CPU will be used, but this may limit compatibility, especially for the interop
        /// functionality, but sometimes also for floating point textures.
        /// </summary>
        /// <returns></returns>
        public static void SelectBestDevice(out ComputePlatform rplatform, out ComputeDevice rdevice) {
            rplatform = null;
            rdevice = null;
            
            var score = -1;
            foreach (var platform in ComputePlatform.Platforms)
            foreach (var device in platform.Devices) {
                var deviceScore = 0;
                if (device.Type == ComputeDeviceTypes.Gpu) {
                    deviceScore += 10;
                    if (!platform.Name.Contains("Intel")) deviceScore += 10;
                }

                if (deviceScore <= score) continue;
                
                rplatform = platform;
                rdevice = device;
                score = deviceScore;
            }
        }

        public static ComputeProgram LoadProgram(string path, ComputeContext context, ComputeDevice device) {
            ComputeProgram program = null;
            
            using (var reader = new StreamReader(path)) {
                var source = reader.ReadToEnd();
                program = new ComputeProgram(context, source);
                try {
                    program.Build(null, null, null, IntPtr.Zero); // compile
                } catch(Exception e) {
                    Console.WriteLine($"Program build log: \n{program.GetBuildLog(device)}");
                    Environment.Exit(1); // TODO: return null
                }
                Console.WriteLine($"Program build log: \n{program.GetBuildLog(device)}"); // log
            }

            return program;
        } 
    }
}