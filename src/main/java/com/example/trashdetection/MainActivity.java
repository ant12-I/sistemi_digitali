package com.example.trashdetection;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity implements View.OnClickListener{
    private static final int SELECT_FILE_GALLERY = 1492;
    private static final int REQUEST_CAMERA = 1861;
    private static final int MY_REQUEST = 100;
    private ImageButton gallery;
    private ImageButton camera;
    private Button detect;
    private ImageView image;
    private Bitmap imgBitmap = null;
    private ResultView mResultView;
    private Module mModule = null;

    private float mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY;

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        gallery = findViewById(R.id.gallery);
        camera = findViewById(R.id.camera);
        image = findViewById(R.id.image);
        detect = findViewById(R.id.detectButton);
        mResultView = findViewById(R.id.resultView);
        mResultView.setVisibility(View.INVISIBLE);
        gallery.setOnClickListener(this);
        camera.setOnClickListener(this);
        detect.setOnClickListener(this);
        detect.setEnabled(false);

        try {
            mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "best.torchscript.ptl"));
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open("classes.txt")));
            String line;
            List<String> classes = new ArrayList<>();
            while ((line = br.readLine()) != null) {
                classes.add(line);
            }
            PrePostProcessor.mClasses = new String[classes.size()];
            classes.toArray(PrePostProcessor.mClasses);
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }

    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.camera:
                if (checkPermission(REQUEST_CAMERA)) {
                    Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(intent, REQUEST_CAMERA);
                } else {
                    requestPermission(REQUEST_CAMERA);
                }
                break;

            case R.id.gallery:
                if (checkPermission(SELECT_FILE_GALLERY)) {
                    Intent intent = new Intent(Intent.ACTION_PICK,
                            MediaStore.Images.Media.INTERNAL_CONTENT_URI);
                    startActivityForResult(intent, SELECT_FILE_GALLERY);
                } else {
                    requestPermission(SELECT_FILE_GALLERY);
                }
                break;

            case R.id.detectButton:

                mImgScaleX = (float) imgBitmap.getWidth() / PrePostProcessor.mInputWidth;
                mImgScaleY = (float) imgBitmap.getHeight() / PrePostProcessor.mInputHeight;

                mIvScaleX = (imgBitmap.getWidth() > imgBitmap.getHeight() ? (float) image.getWidth() / imgBitmap.getWidth() : (float) image.getHeight() / imgBitmap.getHeight());
                mIvScaleY = (imgBitmap.getHeight() > imgBitmap.getWidth() ? (float) image.getHeight() / imgBitmap.getHeight() : (float) image.getWidth() / imgBitmap.getWidth());

                mStartX = (image.getWidth() - mIvScaleX * imgBitmap.getWidth()) / 2;
                mStartY = (image.getHeight() - mIvScaleY * imgBitmap.getHeight()) / 2;
                Bitmap resizedBitmap = Bitmap.createScaledBitmap(imgBitmap, PrePostProcessor.mInputWidth,
                        PrePostProcessor.mInputHeight, true);
                final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap,
                        PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);
                IValue[] outputTuple = mModule.forward(IValue.from(inputTensor)).toTuple();
                final Tensor outputTensor = outputTuple[0].toTensor();
                final float[] outputs = outputTensor.getDataAsFloatArray();
                final ArrayList<Result> results =  PrePostProcessor.outputsToNMSPredictions(outputs,
                        mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY);


                detect.setText(getString(R.string.detect));
                mResultView.setResults(results);
                mResultView.invalidate();
                mResultView.setVisibility(View.VISIBLE);

                break;
        }

    }
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {

        if (requestCode == REQUEST_CAMERA) {
            try {
                imgBitmap = (Bitmap) data.getExtras().get("data");
                imgBitmap = Bitmap.createBitmap(imgBitmap, 0, 0, imgBitmap.getWidth(), imgBitmap.getHeight());
                image.setImageBitmap(imgBitmap);
                detect.setEnabled(true);
            }catch (Exception e){
                Log.e("Object Detection", "no picture taken", e);
            }
        }else if (requestCode == SELECT_FILE_GALLERY){
            try {
                Uri selectedImage = data.getData();
                String[] filePathColumn = {MediaStore.Images.Media.DATA};
                if (selectedImage != null) {
                    Cursor cursor = getContentResolver().query(selectedImage,
                            filePathColumn, null, null, null);
                    if (cursor != null) {
                        cursor.moveToFirst();
                        int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                        String picturePath = cursor.getString(columnIndex);
                        imgBitmap = BitmapFactory.decodeFile(picturePath);
                        imgBitmap = Bitmap.createBitmap(imgBitmap, 0, 0, imgBitmap.getWidth(), imgBitmap.getHeight());
                        image.setImageBitmap(imgBitmap);
                        detect.setEnabled(true);
                        cursor.close();
                    }
                }
            }catch (Exception e){
                Log.e("Object Detection", "no file selected", e);
            }
        }

    }

    public boolean checkPermission(int requestCode) {
        int result = 0;
        if(requestCode ==  REQUEST_CAMERA) {
            result = ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA);
        }else if (requestCode ==  SELECT_FILE_GALLERY ){
            result = ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.READ_EXTERNAL_STORAGE);
        }
        return result == PackageManager.PERMISSION_GRANTED;
    }
    private void requestPermission(int requestCode) {

        if (requestCode == REQUEST_CAMERA) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, MY_REQUEST);
        } else if (requestCode == SELECT_FILE_GALLERY) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, MY_REQUEST);
        }
    }

}

