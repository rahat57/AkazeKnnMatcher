package com.example.ImageProcessingWithGui;

import java.awt.FlowLayout;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;


public class MainClass {
	static{
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}
	static double inlier_threshold =10.0f; // Distance threshold to identify inliers
	static double nn_match_ratio = 0.9f;
	private static Mat H;
	
	private static Mat img1;
	private static Mat img2;
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		 img1=Imgcodecs.imread("Ironmantrain.jpg",0);
//			String path="aloeL.png";
			 img2=Imgcodecs.imread("ironmanquery.png",0);
			
			try {
				System.out.println("Calling AkazeKnnMatcher Function");
				
				//Making List of object because our getAkazeKnnMatcher
				//contain multiple things but at this we are taking only resultant image
				List <Object> getresult=new ArrayList<Object>();
				getresult=getAkazeKnnMatching(img1, img2, 12, 7, 4);
				display(mat2Img((Mat)getresult.get(3)));

			}
			
			catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	}

	// ********NEW FUNCTION//AkazeKnnMAtcher*******
	public static List<Object> getAkazeKnnMatching(Mat img_1, Mat img_2,int detectorType,int extractorType,int matcherType) {
		Long startTime = System.currentTimeMillis();
		List<Object> matchResult=new ArrayList<Object>();
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		Mat homography = new Mat(3, 3, CvType.CV_64FC1);
		homography=getHomoGraphy(img_1, img_2, detectorType, extractorType, matcherType);
	//	System.out.println("values of homography : "+homography.dump());
		FeatureDetector detector = FeatureDetector.create(detectorType);
		MatOfKeyPoint keypoints_object = new MatOfKeyPoint();
		MatOfKeyPoint keypoints_scene = new MatOfKeyPoint();
		detector.detect(img_1, keypoints_object);
		detector.detect(img_2, keypoints_scene);

	
		// -- Step 2: Calculate descriptors (feature vectors)
		DescriptorExtractor extractor = DescriptorExtractor.create(extractorType);
		Mat descriptors_object = new Mat();
		Mat descriptors_scene = new Mat();
		extractor.compute(img_1, keypoints_object, descriptors_object);
		extractor.compute(img_2, keypoints_scene, descriptors_scene);
		
		
		
		// -- Step 3: Matching descriptor vectors using FLANN matcher
		DescriptorMatcher matcher = DescriptorMatcher.create(matcherType);
		
		List<MatOfDMatch> matches=new ArrayList<MatOfDMatch>();
		matcher.knnMatch(descriptors_object, descriptors_scene, matches, 2);
		
		// -- Quick calculation of max and min distances between keypoints
		List<KeyPoint> matched1 = new ArrayList<KeyPoint>();
		List<KeyPoint> matched2 = new ArrayList<KeyPoint>();
		List<KeyPoint> inliers1 = new ArrayList<KeyPoint>();
		List<KeyPoint> inliers2 = new ArrayList<KeyPoint>();
		
		List<DMatch> good_matches = new ArrayList<DMatch>();  //List of the DMatch
		List<KeyPoint> keypoints_objectList=new ArrayList<KeyPoint>();
		keypoints_objectList=keypoints_object.toList();
		List<KeyPoint> keypoints_sceneList=new ArrayList<KeyPoint>();
		keypoints_sceneList=keypoints_scene.toList();
		List<List<DMatch>> nm_matches=new ArrayList<List<DMatch>>();
		
		for(int i=0;i<matches.size();i++)
		{
		
			List<DMatch> temp=matches.get(i).toList();
			nm_matches.add(temp);
			DMatch first = nm_matches.get(i).get(0);
			float dist1 = nm_matches.get(i).get(0).distance;
			float dist2 = nm_matches.get(i).get(1).distance;
			
			if (dist1 < nn_match_ratio * dist2) {
				
				matched1.add(keypoints_objectList.get(first.queryIdx));
				matched2.add(keypoints_sceneList.get(first.trainIdx));
			}
		}
		
		for (int j = 0; j < matched1.size(); j++) {
			
				Mat col = Mat.ones(3, 1, CvType.CV_64F);
			col.put(0, 0, matched1.get(j).pt.x);
			col.put(1, 0, matched1.get(j).pt.y);
//			System.out.println("Values of Matched1 X and Y before multiplying: "+col.dump());
			//Multiplying Col with homography  homography * col;
			Core.gemm(homography, col, 1, new Mat(),1, col);
//			System.out.println("Values of Matched1 X and Y After multiplying: "+col.dump());
			double[] temp=col.get(2,0);
		//	System.out.println("Valu which will devide the col :"+temp[0]);
			for(int r=0;r<col.rows();r++)
			{
				double[] tempMat=col.get(r, 0);
				double data=tempMat[0]/temp[0];
				col.put(r, 0,data);
			//	System.out.println("Values of Matched1 X and Y After Division: "+col.dump());
				
		}
				
				
				 //Solving equation and taking res in dist
			double[] sub1= col.get(0, 0);
			double res1=Math.pow(sub1[0]-matched2.get(j).pt.x, 2);
//			double res1=Math.pow(matched1.get(j).pt.x-matched2.get(j).pt.x, 2);
			double[] subT1=col.get(1, 0);
			double res2=Math.pow((subT1[0]-matched2.get(j).pt.y), 2);
//			double res2=Math.pow((matched1.get(j).pt.y-matched2.get(j).pt.y), 2);
			double dist=Math.sqrt(res1+res2);
			if (dist < inlier_threshold) {
				int new_i = (int) inliers1.size();
				inliers1.add(matched1.get(j));
				inliers2.add(matched2.get(j));
				good_matches.add(new DMatch(new_i, new_i, 0));
				
			}
		}
	
		
		// -- Draw only "good" matches (i.e. whose distance is less than
				// 3*min_dist )
				MatOfDMatch goodmatch = new MatOfDMatch();
				goodmatch.fromList(good_matches);
				Mat img_matches = new Mat();
				MatOfKeyPoint kp = new MatOfKeyPoint();
				kp.fromList(inliers1);
				MatOfKeyPoint pk = new MatOfKeyPoint();
				pk.fromList(inliers2);
				Features2d.drawMatches(img_1, kp, img_2, pk, goodmatch, img_matches);
			//	matchResult.add(img_matches);
				Mat obj_corners=new Mat(4,1,CvType.CV_32FC2);
				Mat scene_corners=new Mat(4,1,CvType.CV_32FC2);
				obj_corners.put(0, 0, new double[]{0,0});
				obj_corners.put(1, 0, new double[]{img_1.cols(),0});
				obj_corners.put(2,0,new double[]{img_1.cols(),img_1.rows()});
				obj_corners.put(3, 0, new double[]{0,img_1.rows()});
				Core.perspectiveTransform(obj_corners, scene_corners, homography);
				//in draw line adding indexes then adding value to new point
				
				Point same=new Point(img_1.cols(), 0);
				//
				Point scene0=new Point(scene_corners.get(0, 0));
			    Point scene1=new Point(scene_corners.get(1, 0));
			    Point scene2=new Point(scene_corners.get(2, 0));
			    Point scene3=new Point(scene_corners.get(3, 0));
			    Point resL1fst=new Point();
			    //Line1 first point
			    resL1fst.x=scene0.x+same.x;
			    resL1fst.y=scene0.y+same.y;
			    //Lin1 2nd point
				Point resL12nd=new Point();
				resL12nd.x=same.x+scene1.x;
				resL12nd.y=same.y+scene1.y;
				// Line2 2nd point
				Point resL22nd=new Point();
				resL22nd.x=same.x+scene2.x;
				resL22nd.y=same.y+scene2.y;
				Point resL32nd=new Point();
				resL32nd.x=same.x+scene3.x;
				resL32nd.y=same.y+scene3.y;
				Imgproc.line(img_matches,resL1fst ,resL12nd, new Scalar(0,255,0), 4);
				Imgproc.line( img_matches,resL12nd,resL22nd, new Scalar(0, 255, 0), 4 );
				Imgproc.line(img_matches,resL22nd,resL32nd, new Scalar(0,255,0), 4);
				Imgproc.line(img_matches,resL32nd,resL1fst , new Scalar(0,255,0), 4);
				
				Long endTime = System.currentTimeMillis();
				Long timeTaken = endTime - startTime;
				matchResult.add(matched1.size()+"");
				double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
				matchResult.add(inliers1.size()+"");
				matchResult.add(timeTaken+"");
				matchResult.add(img_matches);
				System.out.println("A-KAZE Matching Results with BRUTEFORCE_HAMMING");
				System.out.println("*******************************");
				System.out.println("# Keypoints 1: "+keypoints_object.size());
				System.out.println("# Keypoints 2 :"+keypoints_scene.size());
				System.out.println("# Matches :"+matched1.size());
				System.out.println("# Inliers:"+inliers1.size());
				System.out.println("# Inliers Ratio:"+inlier_ratio);
				return matchResult;
	}
	//************FINDING HOMOGRAPHY AT RUNTIME*********************
	public static Mat getHomoGraphy(Mat query,Mat train,int detectorType,int extractorType,int matcherType){

		Long startTime = System.currentTimeMillis();
		
//		System.out.println("query size :"+query.size());
//		System.out.println("train size :"+train.size());
		List<Object> result = new ArrayList<Object>();

		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		

		try{
		FeatureDetector detector = FeatureDetector.create(detectorType);

		final MatOfKeyPoint keypoint1 = new MatOfKeyPoint();
		final MatOfKeyPoint keypoint2 = new MatOfKeyPoint();

		detector.detect(query, keypoint1);
		detector.detect(train, keypoint2);
	//	System.out.println("Key Points1 OF Find Homography" +keypoint1.size());
	//	System.out.println("Key Points2 OF Find Homography" +keypoint2.size());
		// ///////////// Step2 Fearture Descriptor///////////////////////

		DescriptorExtractor extrator = DescriptorExtractor.create(extractorType);
		Mat descriptors1 = new Mat();
		Mat descriptors2 = new Mat();

		extrator.compute(query, keypoint1, descriptors1);
		extrator.compute(train, keypoint2, descriptors2);
	//	System.out.println("descriptors1 OF Find Homography" +descriptors1.size());
	//	System.out.println("descriptors2 OF Find Homography" +descriptors2.size());
		// System.out.println(descriptors1.rows() +" * "+descriptors1.cols());

		// ///////////// Step3 Matching Descriptor with Brute
		// Force///////////////////////

		MatOfDMatch matches = new MatOfDMatch();

		DescriptorMatcher matcher;
		matcher = DescriptorMatcher.create(matcherType);
		
		matcher.match(descriptors1, descriptors2, matches);
	//	System.out.println("Matches  OF Find Homography" +matches.size());
		
		// ////////////////////Step 4 Matching Best
		// Point/////////////////////////////////

		double min_dist = 100;
		double max_dist = 0;

		java.util.List<DMatch> matchesList = matches.toList();

		for (int i = 0; i < descriptors1.rows(); i++) {
			double dist = matchesList.get(i).distance;

			if (dist < min_dist)
				min_dist = dist;
			if (dist > max_dist)
				max_dist = dist;

		}

		LinkedList<DMatch> goodMatches = new LinkedList<DMatch>();

		double sumGoodMatches = 0.0;

		for (int i = 0; i < descriptors1.rows(); i++) {
			if (matchesList.get(i).distance <= 3 * min_dist) {
				goodMatches.add(matchesList.get(i));
				sumGoodMatches += matchesList.get(i).distance;
			}
		}

		double simscore = (double) sumGoodMatches / (double) goodMatches.size();
		result.add(goodMatches.size() + "");
		result.add(simscore + "");
		// ////////////////////////////////////////////////////////

		Mat Img_matches = new Mat();
		MatOfDMatch goodMatchess = new MatOfDMatch();
		goodMatchess.fromList(goodMatches);

		// Features2d.drawMatches(mat, keypoint1,mat2,keypoint2,
		// matches,Img_matches);
		Features2d.drawMatches(query, keypoint1, train, keypoint2,goodMatchess, Img_matches);
	
		List<KeyPoint> keypoint1List=keypoint1.toList();
		List<KeyPoint> keypoin2List=keypoint2.toList();
		LinkedList<Point> keypoint1obj=new LinkedList<Point>();
		LinkedList<Point> kpeypoint2obj=new LinkedList<Point>();
		for(int i=0;i<goodMatches.size();i++){
			
			keypoint1obj.add(keypoint1List.get(goodMatches.get(i).queryIdx).pt);
			kpeypoint2obj.add(keypoin2List.get(goodMatches.get(i).trainIdx).pt);
		}
		MatOfPoint2f key1obj=new MatOfPoint2f();
		MatOfPoint2f key2obj=new MatOfPoint2f();
		key1obj.fromList(keypoint1obj);
		key2obj.fromList(kpeypoint2obj);
		 H=new Mat();
		H=Calib3d.findHomography( key1obj, key2obj, Calib3d.RANSAC,4);
	//	System.out.println("homography matrix Of FindHomography Class:"+H.dump());
		//result.add(H);
		
		Long endTime = System.currentTimeMillis();
		Long timeTaken = endTime - startTime;
		result.add(timeTaken + "");

		result.add(Img_matches);
	
			}catch(Exception e)
		{
			
				JOptionPane.showMessageDialog( null,"OpenCV Error: Bad argument (Specified descriptor extractor type is not supported.)"+ e.getMessage().substring(0, 35));

		}
	//	System.out.println("homography" +H.dump());
		return H;
	}
	//****************FUNCTIONS TO CONVERT Mat2Img and Img2Mat*****************
	public static BufferedImage mat2Img(Mat in) {
		BufferedImage out;
		byte[] data = new byte[in.cols() * in.rows() * (int) in.elemSize()];
		int type;
		in.get(0, 0, data);
		if (in.channels() == 1)
			type = BufferedImage.TYPE_BYTE_GRAY;
		else
			type = BufferedImage.TYPE_3BYTE_BGR;
		out = new BufferedImage(in.cols(), in.rows(), type);
		out.getRaster().setDataElements(0, 0, in.cols(), in.rows(), data);
		return out;
	}

	public static Mat img2Mat(BufferedImage in) {
		Mat out;
		byte[] data;
		int r, g, b;

		out = new Mat(in.getHeight(), in.getWidth(), CvType.CV_8UC3);
		data = new byte[in.getWidth() * in.getHeight() * (int) out.elemSize()];
		int[] dataBuff = in.getRGB(0, 0, in.getWidth(), in.getHeight(), null,
				0, in.getWidth());
		for (int i = 0; i < dataBuff.length; i++) {
			data[i * 3] = (byte) ((dataBuff[i] >> 16) & 0xFF);
			data[i * 3 + 1] = (byte) ((dataBuff[i] >> 8) & 0xFF);
			data[i * 3 + 2] = (byte) ((dataBuff[i] >> 0) & 0xFF);
		}
		out.put(0, 0, data);
		return out;
	}
	
	//*************FUNCTION TO DISPLAY RESULT******************
	public static void display(BufferedImage img) throws IOException
	{
		ImageIcon icon=new ImageIcon(img);
		JFrame frame= new JFrame();
		frame.setLayout(new FlowLayout());
		JLabel lbl=new JLabel();
		lbl.setIcon(icon);
		frame.setSize(900, 500);
		frame.add(lbl);
		frame.setVisible(true);
		frame.setLocationRelativeTo(null);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

	}
}
	

