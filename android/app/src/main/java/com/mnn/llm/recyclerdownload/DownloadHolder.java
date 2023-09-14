package com.mnn.llm.recyclerdownload;

import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.recyclerview.widget.RecyclerView;

import com.mnn.llm.R;

import org.w3c.dom.Text;

import java.util.stream.StreamSupport;

public class DownloadHolder extends RecyclerView.ViewHolder {

    private TextView mModel;
    private ProgressBar mProcess;
    private TextView mPercent;
    private TextView mDownload;
    private Button mButton;

    public DownloadHolder(View v) {
        super(v);
        mModel = v.findViewById(R.id.download_model);
        mProcess = v.findViewById(R.id.download_progress_bar);
        mPercent = v.findViewById(R.id.download_percent);
        mDownload = v.findViewById(R.id.download_size);
        mButton = v.findViewById(R.id.download_again);
    }

    public TextView getName() {
        return mModel;
    }
    public void setName(TextView date) {
        this.mModel = date;
    }

    public ProgressBar getProcessBar() {
        return mProcess;
    }
    public void setProcessBar(ProgressBar date) {
        this.mProcess = date;
    }

    public TextView getPercent() {
        return mPercent;
    }
    public void setPercent(TextView date) {
        this.mPercent = date;
    }

    public TextView getDownload() {
        return mDownload;
    }
    public void setDownload(TextView date) {
        this.mDownload = date;
    }

    public Button getButton() { return mButton; }
}
