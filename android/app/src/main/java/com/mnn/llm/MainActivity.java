package com.mnn.llm;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;

import org.w3c.dom.Text;

import java.io.File;

public class MainActivity extends AppCompatActivity {
    private Chat mChat;
    private Intent mIntent;
    private Button mLoadButton;
    private TextView mModelInfo;
    private RelativeLayout mProcessView;
    private Handler mProcessHandler;
    private ProgressBar mProcessBar;
    private TextView mProcessName;
    private TextView mProcessPercent;
    // resource files
    private String mModelDir = "/data/local/tmp/model";
    private String mTokenizerDir = "";
    private boolean mModelNeedDownload = true;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mIntent = new Intent(this, Conversation.class);
        mModelInfo = (TextView)findViewById(R.id.model_info);
        mLoadButton = (Button)findViewById(R.id.load_button);
        mProcessView = (RelativeLayout)findViewById(R.id.process_view);
        mProcessBar = (ProgressBar)findViewById(R.id.process_bar);
        mProcessName = (TextView)findViewById(R.id.process_name);
        mProcessPercent = (TextView)findViewById(R.id.process_percent);
        mModelDir = this.getCacheDir().toString() + "/model";
        mProcessHandler = new Handler() {
            @Override
            public void handleMessage(Message msg) {
                super.handleMessage(msg);
                int progress = msg.arg1;
                mProcessBar.setProgress((int)progress);
                mProcessPercent.setText(" " + progress + "%");
                if (progress >= 100) {
                    mLoadButton.setClickable(true);
                    mLoadButton.setBackgroundColor(Color.parseColor("#3e3ddf"));
                    mLoadButton.setText("加载已完成");
                    mIntent.putExtra("chat", mChat);
                    startActivity(mIntent);
                }
            }
        };
        /*
        File model = new File(mModelDir, "glm_block_0.mnn");
        if (model.exists()) {
            model.delete();
        }
         */
        onCheckModels();
    }

    @Override
    protected void onResume() {
        super.onResume();
        onCheckModels();
    }

    public void onCheckModels() {
        mModelNeedDownload = checkModelsNeedDownload();
        if (mModelNeedDownload) {
            mModelInfo.setVisibility(View.VISIBLE);
            mModelInfo.setText("使用前请先下载模型！");
            mLoadButton.setText("下载模型");
        } else {
            mModelInfo.setVisibility(View.VISIBLE);
            mModelInfo.setText("模型下载完毕，请加载模型！");
            mLoadButton.setText("加载模型");
        }
    }
    public boolean checkModelsNeedDownload() {
        System.out.println("### Check Models!");
        File dir = new File(mModelDir);
        if (!dir.exists()) {
            return true;
        }
        String[] modelArray = this.getResources().getStringArray(R.array.model_list);
        int[] modelSize = this.getResources().getIntArray(R.array.model_size);
        for (int i = 0; i < modelArray.length; i++) {
            File model = new File(mModelDir, modelArray[i]);
            if (!model.exists()) {
                return true;
            }
            if (model.length() != modelSize[i]) {
                return true;
            }
        }
        return false;
    }

    public void loadModel(View view) {
        if (mModelNeedDownload) {
            startActivity(new Intent(this, DownloadModel.class));
            return;
        }
        mLoadButton.setClickable(false);
        mLoadButton.setBackgroundColor(Color.parseColor("#2454e4"));
        mLoadButton.setText("模型加载中 ...");
        mProcessView.setVisibility(View.VISIBLE);
        mChat = new Chat();
        prepareFiles();
        System.out.println("[MNN_DEBUG] is chat Ready: " + mChat.Ready());
        Handler handler = new Handler() {
            @Override
            public void handleMessage(Message msg) {
                mIntent.putExtra("chat", mChat);
                startActivity(mIntent);
            }
        };
        // copy models
        LoadThread loadT = new LoadThread(mChat, handler, mModelDir, mTokenizerDir);
        loadT.start();
        ProgressThread progressT = new ProgressThread(mChat, mProcessHandler);
        progressT.start();
    }

    public void prepareFiles() {
        System.out.println("MNN_DEBUG: prepareFiles Start");
        try {
            mTokenizerDir = Common.copyAssetResource2File(this, "tokenizer");
        } catch (Exception e) {
            System.out.println(e.toString());
        }
        System.out.println("MNN_DEBUG: prepareFiles End" + mModelDir + " # " + mTokenizerDir);
    }
}

class LoadThread extends Thread {
    private Chat mChat;
    private Handler mHandler;
    private String mModelDir;
    private String mTokenizerDir;
    LoadThread(Chat chat, Handler handler, String modelDir, String tokenizerDir) {
        mChat = chat;
        mHandler = handler;
        mModelDir = modelDir;
        mTokenizerDir = tokenizerDir;
    }
    public void run() {
        super.run();
        mChat.Init(mModelDir, mTokenizerDir);
        mHandler.sendMessage(new Message());
    }
}

class ProgressThread extends Thread {
    private Handler mHandler;
    private Chat mChat;

    ProgressThread(Chat chat, Handler handler) {
        mChat = chat;
        mHandler = handler;
    }

    public void run() {
        super.run();
        float progress = 0;
        while (progress < 100) {
            try {
                Thread.sleep(50);
            } catch (Exception e) {}
            float new_progress = mChat.Progress();
            if (Math.abs(new_progress - progress) < 0.01) {
                continue;
            }
            progress = new_progress;
            Message msg = new Message();
            msg.arg1 = (int)progress;
            mHandler.sendMessage(msg);
        }
    }
}