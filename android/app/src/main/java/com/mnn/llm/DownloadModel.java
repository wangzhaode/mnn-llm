package com.mnn.llm;

import android.app.AlertDialog;
import android.content.DialogInterface;
import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.mnn.llm.recyclerdownload.DownloadRecyclerView;

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
        mAdapter = new DownloadRecyclerView(this, modelArray);
        mRecyclerView.setAdapter(mAdapter);
    }
    public void downloadAll(View view) {
        mDownloadAll.setClickable(false);
        mDownloadAll.setBackgroundColor(Color.parseColor("#2454e4"));
        mDownloadAll.setText("模型下载中 ...");
        mAdapter.onDownload(mDownloadAll);
    }
    public void clearAll(View view) {
        new AlertDialog.Builder(this)
                .setTitle("清空确认")
                .setMessage("确认删除所有已下载模型？")
                .setPositiveButton("确认",
                        new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialog, int which) {
                                mAdapter.onClear();
                            }
                }).show();
    }
}