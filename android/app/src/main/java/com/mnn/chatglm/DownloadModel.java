package com.mnn.chatglm;

import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.view.View;
import android.widget.Button;

import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.mnn.chatglm.recyclerdownload.DownloadRecyclerView;

public class DownloadModel extends BaseActivity {

    private RecyclerView mRecyclerView;
    private DownloadRecyclerView mAdapter;
    private Button mDownloadAll;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_download);
        mRecyclerView = (RecyclerView) findViewById(R.id.download_recycler);
        mRecyclerView.setHasFixedSize(true);
        mRecyclerView.setLayoutManager(new LinearLayoutManager(this));
        mDownloadAll = (Button)findViewById(R.id.download_all);
        // init Data
        String[] modelArray = this.getResources().getStringArray(R.array.model_list);
        int[] modelSize = this.getResources().getIntArray(R.array.model_size);
        mAdapter = new DownloadRecyclerView(this, modelArray, modelSize);
        mRecyclerView.setAdapter(mAdapter);
    }
    public void downloadAll(View view) {
        mDownloadAll.setClickable(false);
        mDownloadAll.setBackgroundColor(Color.parseColor("#2454e4"));
        mDownloadAll.setText("模型下载中 ...");
        mAdapter.onDownload(mDownloadAll);
    }
}