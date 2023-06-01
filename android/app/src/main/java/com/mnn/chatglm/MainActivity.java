package com.mnn.chatglm;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;

import com.mnn.chatglm.Chat;
import com.mnn.chatglm.recylcerchat.ChatData;

import org.w3c.dom.Text;

import java.util.ArrayList;
import java.util.Date;

public class MainActivity extends AppCompatActivity {
    private Chat mChat;
    private Intent mIntent;
    private Button mLoadButton;
    // Progress
    private RelativeLayout mProgressView;
    private Handler mProgressHandler;
    private ProgressBar mProgressBar;
    private TextView mProgressPercent;
    // resource files
    private String mModelDir = "/data/local/tmp/model";
    private String mTokenizerDir = "";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mIntent = new Intent(this, Conversation.class);
        mLoadButton = (Button)findViewById(R.id.load_button);
        mProgressView = (RelativeLayout)findViewById(R.id.progress_view);
        mProgressBar = (ProgressBar)findViewById(R.id.progress_bar);
        mProgressPercent = (TextView)findViewById(R.id.progress_percent);
        mProgressHandler = new Handler() {
            @Override
            public void handleMessage(Message msg) {
                super.handleMessage(msg);
                int progress = (int)(float)msg.obj;
                mProgressBar.setProgress((int)progress);
                mProgressPercent.setText(" " + progress + "%");
                if (progress >= 100) {
                    mLoadButton.setClickable(true);
                    mLoadButton.setBackgroundColor(Color.parseColor("#3e3ddf"));
                    mLoadButton.setText("加载已完成");
                    mIntent.putExtra("chat", mChat);
                    startActivity(mIntent);
                }
            }
        };
    }

    public void runDemo(View view) {
        mLoadButton.setClickable(false);
        mLoadButton.setBackgroundColor(Color.parseColor("#2454e4"));
        mLoadButton.setText("加载中");
        mProgressView.setVisibility(View.VISIBLE);
        System.out.println("[MNN_DEBUG] Running demo...");
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
        ProgressThread progressT = new ProgressThread(mChat, mProgressHandler);
        progressT.start();
    }

    public void prepareFiles() {
        System.out.println("MNN_DEBUG: prepareFiles Start");
        try {
            // mModelDir = Common.copyAssetResource2File(this, "model");
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
            msg.obj = progress;
            mHandler.sendMessage(msg);
        }
    }
}