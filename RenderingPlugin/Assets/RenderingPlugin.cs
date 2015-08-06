using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System.IO;
using System;

public class RenderingPlugin : MonoBehaviour
{
    public static Mesh mesh;
    public string meshURL;

    private static int texSize;
    private static Texture2D tex, nTex;
    private Vector2 oldMousePos;
    public Shader shader;

    private static int UNITY_RENDER_EVENT_ID = 0;
    private static String NORMAL_TEXTURE_ID = "_BumpMap";

    private static int frameCount = 0;
    private static float dt = 0.0f;
    private static float fps = 0.0f;
    private static float updateRate = 4.0f;
    private static Rect rect;
    private static Rect errRect;
    private static string errMessage;

    #region UNUSED
    // private bool isRotating = false;
    private static float ROTATION_SCALE = 5;
    #endregion

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void MyDelegate(string str);

    [DllImport("RenderingPluginDLL")]
    private static extern void UnityRenderEvent(int eventID);
    [DllImport("RenderingPluginDLL")]
    private static extern void Init(int sideSize, IntPtr texPtr, IntPtr nTexPtr, [In, Out] int[] triangles, int trCount);
    [DllImport("RenderingPluginDLL")]
    private static extern void Cleanup();
    [DllImport("RenderingPluginDLL")]
    private static extern void SetDebugFunction(IntPtr fp);
    [DllImport("RenderingPluginDLL")]
    private static extern void ComputeSineWave([In, Out] Vector3[] verts, float time);
    [DllImport("RenderingPluginDLL")]
    private static extern void ParallelComputeSineWave([In, Out] Vector3[] verts, float time);
    [DllImport("RenderingPluginDLL")]
    private static extern void SetTimeFromUnity(float t);

    Vector3[] verts;
    int[] triangles;

    int numVerts;
    int numFaces;

    IEnumerator Start()
    {
        if (meshURL == null)
        {
            Debug.Log("No mesh path selected.");
            Application.Quit();
        }

        if (shader == null)
        {
            Debug.Log("No shader selected.");
            Application.Quit();
        }

        getMeshDataFromFile(meshURL);

        float texSizeF = (float)Math.Sqrt(GetComponent<MeshFilter>().mesh.vertexCount);
        texSize = texSizeF == (int)texSizeF ? (int)texSizeF : (int)texSizeF + 1;

        verts = GetComponent<MeshFilter>().mesh.vertices;

        MyDelegate callbackDelegate = new MyDelegate(callback);
        IntPtr intptrDelegate = Marshal.GetFunctionPointerForDelegate(callbackDelegate);
        SetDebugFunction(intptrDelegate);

        Material material = GetComponent<Renderer>().material;
        material.shader = shader;

        tex = new Texture2D(texSize, texSize, TextureFormat.RGBAFloat, false);
        FillTextureWithData(tex, verts);
        tex.Apply();
        material.mainTexture = tex;

        triangles = mesh.triangles;
        nTex = new Texture2D(texSize, texSize, TextureFormat.RGBAFloat, false);
        FillTextureWithData(nTex, mesh.normals);
        nTex.Apply();

        material.SetTexture(NORMAL_TEXTURE_ID, nTex);

        Init(texSize, tex.GetNativeTexturePtr(), nTex.GetNativeTexturePtr(), mesh.triangles, mesh.triangles.Length);

        rect = new Rect(50, 50, 250, 25);
        errRect = new Rect(50, 75, 250, 100);

        yield return StartCoroutine("CallPluginAtEndOfFrames");
    }

    public void SetMeshURL(string path)
    {
        meshURL = path;
    }

    public void SetShader(string path)
    {
        shader = Resources.Load<Shader>(path);
    }

    void Update()
    {
        #region UNUSED
        /*
        if (Input.GetMouseButtonDown(1))
        {
            isRotating = true;
            oldMousePos = Input.mousePosition;
        }
        else if (Input.GetMouseButtonUp(1)) isRotating = false;
        else if (isRotating) RotateMesh();
        */
        #endregion

        if (Input.GetKeyDown(KeyCode.Escape)) Application.Quit();

        frameCount++;
        dt += Time.deltaTime;
        if (dt > 1.0f / updateRate)
        {
            fps = frameCount / dt;
            frameCount = 0;
            dt -= 1.0f / updateRate;
        }
    }

    void OnGUI()
    {
        GUI.Label(rect, "FPS: " + (int)fps);
        GUI.Label(errRect, errMessage);
    }

    #region UNUSED
    void RotateMesh()
    {
        Vector2 mousePos = new Vector2( Input.mousePosition.x, Input.mousePosition.y );
        Vector2 delta = mousePos - oldMousePos;
        oldMousePos = mousePos;

        transform.Rotate(-Vector3.right * delta.y * Time.deltaTime * ROTATION_SCALE);
        transform.Rotate(Vector3.forward * delta.x * Time.deltaTime * ROTATION_SCALE);
    }
    #endregion

    void OnApplicationQuit()
    {
        Cleanup();
    }

    void getMeshDataFromFile(string url)
    {
        mesh = new Mesh();

        getVertsFromFile(url);
        getFacesFromFile(url);

        mesh.vertices = verts;
        mesh.triangles = triangles;

        setMeshUVs();

        GetComponent<MeshFilter>().mesh = mesh;
    }

    void getVertsFromFile(string url) 
    {
        numVerts = 0;

        string[] lines = File.ReadAllLines(url);
        foreach(string line in lines)
        {
            string[] tokens = line.Split(' ');
            if (tokens[0] == "v")
            {
                numVerts++;
            }
        }

        int index = 0;

        verts = new Vector3[numVerts];

        foreach (string line in lines)
        {
            string[] tokens = line.Split(' ');
            if (tokens[0] == "v")
            {
                verts[index++] = new Vector3(float.Parse(tokens[1]), float.Parse(tokens[2]), float.Parse(tokens[3]));
            }
        }
    }

    void setMeshUVs()
    {
        Vector2[] uvs = new Vector2[mesh.vertices.Length];
        int texSide = (int)Math.Sqrt(uvs.Length);

        for (int i = 0; i < uvs.Length; i++)
        {
            int row = i / texSide;
            int col = i % texSide;

            uvs[i] = new Vector2
                (((float)row / texSide + (float)(row + 1) / texSide) / 2.0f,
                ((float)col / texSide + (float)(col + 1) / texSide) / 2.0f);
        }

        mesh.uv = uvs;
    }

    void getFacesFromFile(string url)
    {
        numFaces = 0;

        string[] lines = File.ReadAllLines(url);
        foreach (string line in lines)
        {
            string[] tokens = line.Split(' ');
            if (tokens[0] == "f")
            {
                numFaces++;
            }
        }

        triangles = new int[numFaces * 3];

        int index = 0;

        foreach (string line in lines)
        {
            string[] tokens = line.Split(' ');
            if (tokens[0] == "f")
            {
                triangles[index++] = int.Parse(tokens[1]) - 1;
                triangles[index++] = int.Parse(tokens[2]) - 1;
                triangles[index++] = int.Parse(tokens[3]) - 1;
            }
        }
    }

    static void callback(string str)
    {
        errMessage = str;
        // Debug.Log("Callback: " + str);
    }

    byte[] getBytesFromVector3Array(Vector3[] array)
    {
        byte[] bytes = new byte[array.Length * 4 * 4];

        int index = 0;
        float one = 1.0f;

        for (int i = 0; i < array.Length; i++)
        {
            Buffer.BlockCopy(BitConverter.GetBytes(array[i].x), 0, bytes, index, 4);
            index += 4;
            Buffer.BlockCopy(BitConverter.GetBytes(array[i].y), 0, bytes, index, 4);
            index += 4;
            Buffer.BlockCopy(BitConverter.GetBytes(array[i].z), 0, bytes, index, 4);
            index += 4;
            Buffer.BlockCopy(BitConverter.GetBytes(one), 0, bytes, index, 4);
            index += 4;
        }

        return bytes;
    }

    private void FillTextureWithData(Texture2D tex, Vector3[] vertices)
    {
        for(int i = 0; i < vertices.Length; i++){
            Vector3 v = vertices[i];
            Color color = new Color(v[0], v[1], v[2], i);
            tex.SetPixel(i / texSize, i % texSize, color);
        }
    }

    private IEnumerator CallPluginAtEndOfFrames()
    {
        while (true)
        {
            yield return new WaitForEndOfFrame();

            SetTimeFromUnity(Time.timeSinceLevelLoad);

            GL.IssuePluginEvent(UNITY_RENDER_EVENT_ID);
        }
    }
}