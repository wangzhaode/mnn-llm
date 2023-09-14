package com.mnn.llm.recyclerdownload;

import android.os.Handler;
import android.os.Message;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

public class DownloadData {
    Handler mHandler;
    String mDir;
    String mName;
    int mIdx;
    int mProcess;
    int mDownload;
    int mTotal;
    int mSuccess;

    public DownloadData(Handler handler, String dir, String name, int idx, int total) {
        mHandler = handler;
        mDir = dir;
        mName = name;
        mIdx = idx;
        mTotal = total;
        mProcess = 0;
        mDownload = 0;
        mSuccess = 0;
    }

    public String getName() {
        return mName;
    }

    public void setName(String name) {
        this.mName = name;
    }

    public int getIdx() {
        return mIdx;
    }

    public void setIdx(int idx) {
        this.mIdx = idx;
    }

    public int getProcess() {
        return mProcess;
    }

    public void setProcess(int process) {
        this.mProcess = process;
    }

    public int getDownload() {
        return mDownload;
    }

    public void setDownload(int process) {
        this.mDownload = process;
    }

    public int getTotal() {
        return mTotal;
    }

    public void setTotal(int process) {
        this.mTotal = process;
    }

    public int getSuccess() {
        return mSuccess;
    }

    public void setSuccess(int sucess) { this.mSuccess = sucess; }

    public void onClear() {
        File file = new File(mDir, mName);
        if (file.exists()) {
            file.delete();
        }
    }
    public void onDownload(int nextDownloadIdx) {
        // check has download
        File file = new File(mDir, mName);
        if (file.exists()) {
            if (mTotal == (int) file.length()) {
                mProcess = 100;
                mDownload = mTotal;
                mSuccess = 1;
                Message msg = new Message();
                msg.what = nextDownloadIdx;
                mHandler.sendMessage(msg);
                return;
            }
        }
        final String url = "https://huggingface.co/zhaode/llm-mnn/resolve/main/" + mName;
        final long startTime = System.currentTimeMillis();
        OkHttpClient okHttpClient = new OkHttpClient();

        Request request = new Request.Builder().url(url).build();
        okHttpClient.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                // 下载失败
                e.printStackTrace();
                Log.i("DOWNLOAD","download failed");
                mSuccess = -1;
                if (nextDownloadIdx >= 0) {
                    Message msg = new Message();
                    msg.what = nextDownloadIdx;
                    mHandler.sendMessage(msg);
                }
            }
            @Override
            public void onResponse(Call call, Response response) throws IOException {
                InputStream is = null;
                byte[] buf = new byte[2048];
                int len = 0;
                FileOutputStream fos = null;
                try {
                    is = response.body().byteStream();
                    long total = response.body().contentLength();
                    File file = new File(mDir, mName);
                    fos = new FileOutputStream(file);
                    long sum = 0;
                    while ((len = is.read(buf)) != -1) {
                        fos.write(buf, 0, len);
                        sum += len;
                        int progress = (int) (sum * 1.0f / total * 100);
                        mProcess = progress;
                        mDownload = (int)sum;
                        mTotal = (int)total;
                        Message msg = new Message();
                        msg.what = -1;
                        mHandler.sendMessage(msg);
                    }
                    fos.flush();
                    // 下载完成
                    // listener.onDownloadSuccess();
                    Log.i("DOWNLOAD","download success");
                    mSuccess = 1;
                    if (nextDownloadIdx >= 0) {
                        Message msg = new Message();
                        msg.what = nextDownloadIdx;
                        mHandler.sendMessage(msg);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                    // listener.onDownloadFailed();
                    Log.i("DOWNLOAD","download failed");
                    mSuccess = -1;
                } finally {
                    try {
                        if (is != null)
                            is.close();
                    } catch (IOException e) {
                        mSuccess = -1;
                    }
                    try {
                        if (fos != null)
                            fos.close();
                    } catch (IOException e) {
                        mSuccess = -1;
                    }
                }
            }
        });
    }
}
