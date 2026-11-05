#pragma once

#include "STrack.h"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    vector<float> feature;
};

class BYTETracker
{
public:
	BYTETracker(float high, float new_track, float match, double lost_ms);
	~BYTETracker();

	vector<STrack> update(const vector<Object>& objects, double timestamp_ms);
    const vector<STrack>& tracked() const { return tracked_stracks; }
    const vector<STrack>& lost() const { return lost_stracks; }
	Scalar get_color(int idx);

private:
    void appearance_distance(vector<vector<float>>& costs, const vector<STrack*>& tracks, const vector<STrack>& detections);
	vector<STrack*> joint_stracks(vector<STrack*> &tlista, vector<STrack> &tlistb);
	vector<STrack> joint_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);

	vector<STrack> sub_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);
	void remove_duplicate_stracks(vector<STrack> &resa, vector<STrack> &resb, vector<STrack> &stracksa, vector<STrack> &stracksb);

	void linear_assignment(vector<vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
		vector<vector<int> > &matches, vector<int> &unmatched_a, vector<int> &unmatched_b);
	vector<vector<float> > iou_distance(vector<STrack*> &atracks, vector<STrack> &btracks, int &dist_size, int &dist_size_size);
	vector<vector<float> > iou_distance(vector<STrack> &atracks, vector<STrack> &btracks);
	vector<vector<float> > ious(vector<vector<float> > &atlbrs, vector<vector<float> > &btlbrs);

	double lapjv(const vector<vector<float> > &cost, vector<int> &rowsol, vector<int> &colsol, 
		bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:

	float track_thresh;
	float high_thresh;
	float match_thresh;
	int frame_id;
	double max_time_lost;

	vector<STrack> tracked_stracks;
	vector<STrack> lost_stracks;
	byte_kalman::KalmanFilter kalman_filter;
};