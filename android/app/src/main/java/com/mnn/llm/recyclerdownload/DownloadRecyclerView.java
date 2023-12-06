package com.mnn.llm.recyclerdownload;

import android.content.Context;
import android.graphics.Color;
import android.os.Handler;
import android.os.Message;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.Toast;

import androidx.recyclerview.widget.RecyclerView;

import com.mnn.llm.R;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DownloadRecyclerView extends RecyclerView.Adapter<RecyclerView.ViewHolder> {
    private List<DownloadData> mItems;
    private Context mContext;
    private Handler mHandler;
    private Button mButton;

    public DownloadRecyclerView(Context context, String[] models) {
        this.mContext = context;
        this.mItems = new ArrayList<DownloadData>();
        final String modelDir = context.getCacheDir().toString() + "/model";
        File modelPath = new File(modelDir);
        if (!modelPath.exists()) {
            modelPath.mkdirs();
        }
        mHandler = new Handler() {
            @Override
            public void handleMessage(Message msg) {
                super.handleMessage(msg);
                notifyDataSetChanged();
                if (msg.what >= 0 && msg.what < mItems.size()) {
                    mItems.get(msg.what).onDownload(msg.what + 1);
                }
                if (msg.what >= 0 && msg.what == mItems.size()) {
                    mButton.setText("下载结束");
                    mButton.setBackgroundColor(Color.parseColor("#2454e4"));
                }
            }
        };
        for (int i = 0; i < models.length; i++) {
            this.mItems.add(new DownloadData(mHandler, modelDir, models[i], i, 25751300));
        }
    }

    // Return the size of your dataset (invoked by the layout manager)
    @Override
    public int getItemCount() {
        return this.mItems.size();
    }

    @Override
    public int getItemViewType(int position) {
        return mItems.get(position).getIdx();
    }

    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(ViewGroup viewGroup, int viewType) {
        LayoutInflater inflater = LayoutInflater.from(viewGroup.getContext());
        View v = inflater.inflate(R.layout.download_item, viewGroup, false);
        RecyclerView.ViewHolder viewHolder = new DownloadHolder(v);
        return viewHolder;
    }
    @Override
    public void onBindViewHolder(RecyclerView.ViewHolder viewHolder, int position) {
        DownloadHolder vh = (DownloadHolder) viewHolder;
        DownloadData item = mItems.get(position);
        vh.getName().setText(item.getName());
        if (item.getSuccess() >= 0) {
            vh.getPercent().setText(item.getProcess() + "%");
            vh.getProcessBar().setProgress(item.getProcess());
            float download = (float) (item.getDownload() / 1024.0 / 1024.0);
            float total = (float) (item.getTotal() / 1024.0 / 1024.0);
            vh.getDownload().setText(String.format("%.2f M / %.2f M", download, total));
        } else {
            vh.getDownload().setText("下载失败, 请点击左侧logo重新下载");
        }
        vh.getButton().setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                vh.getDownload().setText("尝试重新下载");
                mItems.get(position).setSuccess(0);
                mItems.get(position).onDownload(-1);
            }
        });
    }

    public void onDownload(View view) {
        mButton = (Button)view;
        // download item 0 and next download 1
        mItems.get(0).onDownload(1);
    }

    public void onClear() {
        for (DownloadData item : mItems) {
            item.onClear();
        }
        Toast.makeText(this.mContext, "已清空所有模型文件", Toast.LENGTH_SHORT).show();
    }
}
