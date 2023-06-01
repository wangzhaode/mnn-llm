package com.mnn.chatglm;

import android.content.res.AssetManager;

import java.io.Serializable;
import java.util.ArrayList;

public class Chat implements Serializable {
    public native boolean Init(String modelDir, String tokenizerDir);
    public native boolean Ready();
    public native float Progress();
    public native String Submit(String input);
    public native String Response();
    public native void Done();
    public native void Reset();

    static {
        System.loadLibrary("chatglm_mnn");
    }
}
