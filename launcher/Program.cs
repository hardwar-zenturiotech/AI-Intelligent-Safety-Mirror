using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Windows.Forms;

namespace SafetyMirror
{
    public static class Program
    {
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool AllocConsole();

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool FreeConsole();

        [DllImport("user32.dll")]
        private static extern bool SetForegroundWindow(IntPtr hWnd);

        private static HttpListener httpListener;
        private static Process emoshProcess;
        private static string baseDirectory;
        private static string distDirectory;
        private static string recordingsDirectory;
        private static int serverPort = 8088;
        private static bool isRunning = true;
        private static bool verboseLogging = false;

        private static readonly Dictionary<string, string> MimeTypes = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            { ".html", "text/html; charset=utf-8" },
            { ".htm", "text/html; charset=utf-8" },
            { ".js", "text/javascript; charset=utf-8" },
            { ".mjs", "text/javascript; charset=utf-8" },
            { ".css", "text/css; charset=utf-8" },
            { ".json", "application/json; charset=utf-8" },
            { ".wasm", "application/wasm" },
            { ".glb", "model/gltf-binary" },
            { ".task", "application/octet-stream" },
            { ".onnx", "application/octet-stream" },
            { ".fbx", "application/octet-stream" },
            { ".png", "image/png" },
            { ".jpg", "image/jpeg" },
            { ".jpeg", "image/jpeg" },
            { ".gif", "image/gif" },
            { ".svg", "image/svg+xml" },
            { ".ico", "image/x-icon" },
            { ".webm", "video/webm" },
            { ".mp4", "video/mp4" },
            { ".map", "application/json" }
        };

        [STAThread]
        public static int Main(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            // Determine if console logging requested
            foreach (var arg in args)
            {
                if (arg.Equals("--console", StringComparison.OrdinalIgnoreCase) ||
                    arg.Equals("-c", StringComparison.OrdinalIgnoreCase) ||
                    arg.Equals("/debug", StringComparison.OrdinalIgnoreCase))
                {
                    verboseLogging = true;
                    AllocConsole();
                    Console.Title = "AI Safety Mirror - System Console";
                    break;
                }
            }

            Log("=========================================================");
            Log("     AI INTELLIGENT SAFETY MIRROR DESKTOP RUNNER         ");
            Log("=========================================================");

            baseDirectory = AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\', '/');

            // Look for supporting files folder (resources, app_data, dist, frontend/dist)
            string[] candidateDirs = new string[]
            {
                Path.Combine(baseDirectory, "resources"),
                Path.Combine(baseDirectory, "app_data"),
                Path.Combine(baseDirectory, "supporting_files"),
                Path.Combine(baseDirectory, "dist"),
                Path.Combine(baseDirectory, "frontend", "dist")
            };

            distDirectory = null;
            foreach (var dir in candidateDirs)
            {
                if (Directory.Exists(dir))
                {
                    string testIndex = Path.Combine(dir, "index.html");
                    string testNested = Path.Combine(dir, "frontend", "index.html");
                    if (File.Exists(testIndex) || File.Exists(testNested))
                    {
                        distDirectory = dir;
                        break;
                    }
                }
            }
            if (distDirectory == null)
            {
                distDirectory = Path.Combine(baseDirectory, "resources");
            }

            recordingsDirectory = Path.Combine(baseDirectory, "recordings");
            if (!Directory.Exists(recordingsDirectory))
            {
                try { Directory.CreateDirectory(recordingsDirectory); } catch { }
            }

            // Verify production build assets exist
            string indexFile = Path.Combine(distDirectory, "index.html");
            if (!File.Exists(indexFile))
            {
                // Check if nested in dist/frontend
                string nestedIndex = Path.Combine(distDirectory, "frontend", "index.html");
                if (File.Exists(nestedIndex))
                {
                    try { File.Copy(nestedIndex, indexFile, true); } catch { }
                }
            }

            if (!File.Exists(indexFile))
            {
                MessageBox.Show(
                    "Production assets not found at:\n" + distDirectory + "\n\n" +
                    "Please run 'npm run build' inside the frontend directory before running the desktop application.",
                    "AI Safety Mirror - Build Required",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
                return 1;
            }

            // Parse custom port if provided
            for (int i = 0; i < args.Length; i++)
            {
                if ((args[i] == "--port" || args[i] == "-p") && i + 1 < args.Length)
                {
                    int p;
                    if (int.TryParse(args[i + 1], out p)) serverPort = p;
                }
            }

            // Start Embedded Web Server
            if (!StartWebServer())
            {
                MessageBox.Show(
                    "Unable to start local web server on port " + serverPort + ".\n" +
                    "Please ensure no other application is blocking local network ports.",
                    "AI Safety Mirror - Port Conflict",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
                return 2;
            }

            // Optionally Start EMOSH Python Microservice in background
            bool skipEmosh = false;
            foreach (var arg in args)
            {
                if (arg.Equals("--no-emosh", StringComparison.OrdinalIgnoreCase))
                {
                    skipEmosh = true;
                    break;
                }
            }

            if (!skipEmosh)
            {
                StartEmoshMicroservice();
            }

            // Detect Chrome or Edge
            string browserPath = FindSupportedBrowser(args);
            if (string.IsNullOrEmpty(browserPath))
            {
                MessageBox.Show(
                    "Google Chrome or Microsoft Edge was not found on your system.\n\n" +
                    "Please install Google Chrome or Microsoft Edge to run the AI Intelligent Safety Mirror.",
                    "AI Safety Mirror - Browser Required",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Warning);

                // Fallback to opening default system browser
                Process.Start("http://127.0.0.1:" + serverPort + "/");
                return 0;
            }

            Log("Found browser engine: " + browserPath);

            // Prepare isolated user profile directory
            string profileDir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "SafetyMirrorApp",
                "Profile");

            try
            {
                if (!Directory.Exists(profileDir))
                {
                    Directory.CreateDirectory(profileDir);
                }
            }
            catch { }

            string appUrl = string.Format("http://127.0.0.1:{0}/", serverPort);

            // Construct browser launch arguments for peak GPU acceleration & clean window mode
            StringBuilder sb = new StringBuilder();
            sb.AppendFormat("--app=\"{0}\" ", appUrl);
            sb.AppendFormat("--user-data-dir=\"{0}\" ", profileDir);
            sb.Append("--window-size=1440,900 ");
            sb.Append("--window-position=center ");
            sb.Append("--enable-gpu-rasterization ");
            sb.Append("--enable-zero-copy ");
            sb.Append("--enable-webgl ");
            sb.Append("--enable-webgl2-compute-context ");
            sb.Append("--enable-features=Vulkan,DefaultANGLEVulkan,WebGPU ");
            sb.Append("--use-angle=d3d11 ");
            sb.Append("--ignore-gpu-blocklist ");
            sb.Append("--disable-pinch ");
            sb.Append("--autoplay-policy=no-user-gesture-required ");
            sb.Append("--disable-features=Translate ");
            sb.AppendFormat("--unsafely-treat-insecure-origin-as-secure=\"{0}\" ", appUrl);

            foreach (var arg in args)
            {
                if (arg.Equals("--kiosk", StringComparison.OrdinalIgnoreCase))
                {
                    sb.Append("--kiosk ");
                }
            }

            Log("Launching standalone application window at " + appUrl);

            ProcessStartInfo psi = new ProcessStartInfo();
            psi.FileName = browserPath;
            psi.Arguments = sb.ToString();
            psi.UseShellExecute = false;

            Process browserProcess = null;
            try
            {
                browserProcess = Process.Start(psi);
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "Error launching desktop application:\n" + ex.Message,
                    "AI Safety Mirror - Launch Error",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
                StopAll();
                return 3;
            }

            // Wait for user to close the browser window
            if (browserProcess != null)
            {
                Log("Application running. Waiting for user exit...");
                try
                {
                    browserProcess.WaitForExit();
                }
                catch { }
            }

            Log("Browser closed by user. Cleaning up background services...");
            StopAll();
            Log("Application terminated cleanly.");
            return 0;
        }

        private static bool StartWebServer()
        {
            int maxAttempts = 10;
            for (int attempt = 0; attempt < maxAttempts; attempt++)
            {
                try
                {
                    httpListener = new HttpListener();
                    httpListener.Prefixes.Add(string.Format("http://127.0.0.1:{0}/", serverPort));
                    httpListener.Start();
                    Log(string.Format("Embedded HTTP Server running at http://127.0.0.1:{0}/", serverPort));
                    ThreadPool.QueueUserWorkItem(ListenLoop);
                    return true;
                }
                catch (HttpListenerException)
                {
                    Log(string.Format("Port {0} is occupied; trying next port {1}...", serverPort, serverPort + 1));
                    serverPort++;
                    try { if (httpListener != null) httpListener.Close(); } catch { }
                }
                catch (Exception ex)
                {
                    Log("Web server startup exception: " + ex.Message);
                    return false;
                }
            }
            return false;
        }

        private static void ListenLoop(object state)
        {
            while (isRunning && httpListener != null && httpListener.IsListening)
            {
                try
                {
                    var context = httpListener.GetContext();
                    ThreadPool.QueueUserWorkItem(HandleRequest, context);
                }
                catch (HttpListenerException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    if (isRunning) Log("Listener error: " + ex.Message);
                }
            }
        }

        private static void HandleRequest(object state)
        {
            HttpListenerContext context = state as HttpListenerContext;
            if (context == null) return;

            HttpListenerRequest request = context.Request;
            HttpListenerResponse response = context.Response;

            try
            {
                // Standard CORS & Web Security Headers
                response.Headers["Access-Control-Allow-Origin"] = "*";
                response.Headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, HEAD";
                response.Headers["Access-Control-Allow-Headers"] = "Content-Type, Range";
                response.Headers["X-Content-Type-Options"] = "nosniff";
                response.Headers["Cross-Origin-Opener-Policy"] = "same-origin";
                response.Headers["Cross-Origin-Embedder-Policy"] = "require-corp";

                if (request.HttpMethod == "OPTIONS")
                {
                    response.StatusCode = 204;
                    response.Close();
                    return;
                }

                string rawPath = request.Url.LocalPath;
                string cleanPath = rawPath.TrimStart('/');

                // Health check
                if (cleanPath.Equals("api/server-health", StringComparison.OrdinalIgnoreCase))
                {
                    byte[] data = Encoding.UTF8.GetBytes(string.Format(
                        "{{\"status\":\"ok\",\"app\":\"AI Intelligent Safety Mirror\",\"port\":{0}}}",
                        serverPort));
                    response.ContentType = "application/json; charset=utf-8";
                    response.StatusCode = 200;
                    response.ContentLength64 = data.Length;
                    response.OutputStream.Write(data, 0, data.Length);
                    response.Close();
                    return;
                }

                // Handle video recordings upload
                if (cleanPath.Equals("api/recordings", StringComparison.OrdinalIgnoreCase) &&
                    request.HttpMethod == "POST")
                {
                    HandleRecordingUpload(request, response);
                    return;
                }

                // Normalize file path for SPA & static serving
                string relativePath = cleanPath;
                if (string.IsNullOrEmpty(relativePath) ||
                    relativePath.Equals("frontend", StringComparison.OrdinalIgnoreCase) ||
                    relativePath.Equals("frontend/", StringComparison.OrdinalIgnoreCase) ||
                    relativePath.Equals("index.html", StringComparison.OrdinalIgnoreCase))
                {
                    relativePath = "index.html";
                }
                else if (relativePath.StartsWith("frontend/", StringComparison.OrdinalIgnoreCase))
                {
                    relativePath = relativePath.Substring("frontend/".Length);
                    if (string.IsNullOrEmpty(relativePath)) relativePath = "index.html";
                }

                string filePath = Path.Combine(distDirectory, relativePath.Replace('/', '\\'));

                // If not found, fallback to index.html (SPA routing support)
                if (!File.Exists(filePath))
                {
                    string fallbackIndex = Path.Combine(distDirectory, "index.html");
                    if (File.Exists(fallbackIndex))
                    {
                        filePath = fallbackIndex;
                    }
                    else
                    {
                        response.StatusCode = 404;
                        byte[] err = Encoding.UTF8.GetBytes("404 - Asset Not Found");
                        response.OutputStream.Write(err, 0, err.Length);
                        response.Close();
                        return;
                    }
                }

                string ext = Path.GetExtension(filePath);
                string contentType;
                if (!MimeTypes.TryGetValue(ext, out contentType))
                {
                    contentType = "application/octet-stream";
                }

                response.ContentType = contentType;
                response.StatusCode = 200;

                FileInfo fi = new FileInfo(filePath);
                response.ContentLength64 = fi.Length;

                if (ext.Equals(".wasm", StringComparison.OrdinalIgnoreCase) ||
                    ext.Equals(".task", StringComparison.OrdinalIgnoreCase) ||
                    ext.Equals(".onnx", StringComparison.OrdinalIgnoreCase) ||
                    ext.Equals(".glb", StringComparison.OrdinalIgnoreCase))
                {
                    response.Headers["Cache-Control"] = "public, max-age=31536000, immutable";
                }

                using (FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read))
                {
                    byte[] buffer = new byte[64 * 1024];
                    int read;
                    while ((read = fs.Read(buffer, 0, buffer.Length)) > 0)
                    {
                        response.OutputStream.Write(buffer, 0, read);
                    }
                }
                response.Close();
            }
            catch (Exception ex)
            {
                try
                {
                    response.StatusCode = 500;
                    byte[] err = Encoding.UTF8.GetBytes("Internal Server Error: " + ex.Message);
                    response.OutputStream.Write(err, 0, err.Length);
                    response.Close();
                }
                catch { }
            }
        }

        private static void HandleRecordingUpload(HttpListenerRequest request, HttpListenerResponse response)
        {
            try
            {
                string contentType = request.ContentType ?? "";
                bool isMp4 = contentType.IndexOf("mp4", StringComparison.OrdinalIgnoreCase) >= 0;
                string ext = isMp4 ? "mp4" : "webm";

                string filename = string.Format(
                    "safety_mirror_{0}_{1}.{2}",
                    DateTime.Now.ToString("yyyy-MM-ddTHH-mm-ss"),
                    Guid.NewGuid().ToString("N").Substring(0, 8),
                    ext);

                string finalPath = Path.Combine(recordingsDirectory, filename);
                string tempPath = Path.Combine(recordingsDirectory, "." + filename + ".part");

                long bytesWritten = 0;
                using (FileStream fs = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None))
                {
                    byte[] buffer = new byte[64 * 1024];
                    int read;
                    while ((read = request.InputStream.Read(buffer, 0, buffer.Length)) > 0)
                    {
                        fs.Write(buffer, 0, read);
                        bytesWritten += read;
                    }
                }

                if (File.Exists(finalPath)) File.Delete(finalPath);
                File.Move(tempPath, finalPath);

                string json = string.Format("{{\"filename\":\"{0}\",\"bytesWritten\":{1}}}", filename, bytesWritten);
                byte[] data = Encoding.UTF8.GetBytes(json);

                response.StatusCode = 201;
                response.ContentType = "application/json; charset=utf-8";
                response.ContentLength64 = data.Length;
                response.OutputStream.Write(data, 0, data.Length);
                response.Close();
                Log("Saved recording: " + filename + " (" + (bytesWritten / 1024) + " KB)");
            }
            catch (Exception ex)
            {
                response.StatusCode = 500;
                byte[] err = Encoding.UTF8.GetBytes("{\"error\":\"" + ex.Message.Replace("\"", "\\\"") + "\"}");
                response.ContentType = "application/json";
                response.OutputStream.Write(err, 0, err.Length);
                response.Close();
            }
        }

        private static void StartEmoshMicroservice()
        {
            try
            {
                string[] serverCandidates = new string[]
                {
                    Path.Combine(baseDirectory, "resources", "emosh_service", "server.py"),
                    Path.Combine(baseDirectory, "emosh_service", "server.py")
                };
                string serverScript = null;
                foreach (var s in serverCandidates)
                {
                    if (File.Exists(s)) { serverScript = s; break; }
                }
                if (serverScript == null) return;

                string venvPython = Path.Combine(Path.GetDirectoryName(serverScript), ".venv", "Scripts", "python.exe");
                string pythonExe = File.Exists(venvPython) ? venvPython : "python.exe";

                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = pythonExe;
                psi.Arguments = string.Format("\"{0}\" 8000", serverScript);
                psi.WorkingDirectory = baseDirectory;
                psi.CreateNoWindow = true;
                psi.UseShellExecute = false;

                emoshProcess = Process.Start(psi);
                Log("Launched EMOSH microservice on port 8000 (PID: " + (emoshProcess != null ? emoshProcess.Id : 0) + ")");
            }
            catch (Exception ex)
            {
                Log("EMOSH microservice could not be started: " + ex.Message);
            }
        }

        private static string FindSupportedBrowser(string[] args)
        {
            // Allow user to force chrome or edge via flag
            bool forceChrome = false;
            bool forceEdge = false;
            foreach (var arg in args)
            {
                if (arg.Equals("--chrome", StringComparison.OrdinalIgnoreCase)) forceChrome = true;
                if (arg.Equals("--edge", StringComparison.OrdinalIgnoreCase)) forceEdge = true;
            }

            string[] chromePaths = new string[]
            {
                @"C:\Program Files\Google\Chrome\Application\chrome.exe",
                @"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Google\Chrome\Application\chrome.exe")
            };

            string[] edgePaths = new string[]
            {
                @"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                @"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Microsoft\Edge\Application\msedge.exe")
            };

            if (forceChrome)
            {
                foreach (var p in chromePaths) if (File.Exists(p)) return p;
            }
            if (forceEdge)
            {
                foreach (var p in edgePaths) if (File.Exists(p)) return p;
            }

            // Prefer Chrome for optimal WebGPU/WebGL, fallback to Edge
            foreach (var p in chromePaths) if (File.Exists(p)) return p;
            foreach (var p in edgePaths) if (File.Exists(p)) return p;

            return null;
        }

        private static void StopAll()
        {
            isRunning = false;

            if (httpListener != null)
            {
                try
                {
                    httpListener.Stop();
                    httpListener.Close();
                }
                catch { }
                httpListener = null;
            }

            if (emoshProcess != null)
            {
                try
                {
                    if (!emoshProcess.HasExited)
                    {
                        emoshProcess.Kill();
                    }
                }
                catch { }
                emoshProcess = null;
            }
        }

        private static void Log(string message)
        {
            if (verboseLogging)
            {
                Console.WriteLine("[{0}] {1}", DateTime.Now.ToString("HH:mm:ss"), message);
            }
        }
    }
}
