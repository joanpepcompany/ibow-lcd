/**
* This file is part of ibow-lcd.
*
* Copyright (C) 2017 Emilio Garcia-Fidalgo <emilio.garcia@uib.es> (University of the Balearic Islands)
*
* ibow-lcd is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ibow-lcd is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ibow-lcd. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef INCLUDE_IBOW_LCD_LCDETECTOR_H_
#define INCLUDE_IBOW_LCD_LCDETECTOR_H_

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <queue>
#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <boost/filesystem.hpp>

#include "ibow-lcd/island.h"
#include "ibow-lcd/geometricconstraints.h"
#include "obindex2/binary_index.h"
#include "Logos.h"
#include "lpm.h"
#include "gms_matcher.h"

#include <future>
#include <thread>

using namespace std;

namespace ibow_lcd
{

// LCDetectorParams
struct LCDetectorParams
{
  LCDetectorParams() : k(16),
                       s(150),
                       t(4),
                       merge_policy(obindex2::MERGE_POLICY_NONE),
                       purge_descriptors(true),
                       min_feat_apps(2),
                       p(250),
                       nndr_bf_lines(0.7),
                       nndr(0.8f),
                       nndr_bf(0.8f),
                       ep_dist(2.0),
                       conf_prob(0.985),
                       min_score(0.3),
                       island_size(7),
                       min_inliers(22),
                       nframes_after_lc(3),
                       min_consecutive_loops(5),
                       non_sim_candidates(0.5f),
                       debug_loops(false),
                       b_l_endpts(false),
                       b_l_inters_pts(false),
                       b_l_global_rot(false),
                       b_l_center_pts(false),
                       detect_grad_th(50)
  {
  }

  int detect_grad_th;
  // Image index params
  unsigned k;                         // Branching factor for the image index
  unsigned s;                         // Maximum leaf size for the image index
  unsigned t;                         // Number of trees to search in parallel
  obindex2::MergePolicy merge_policy; // Merging policy
  bool purge_descriptors;             // Delete descriptors from index?
  unsigned min_feat_apps;             // Min apps of a feature to be a visual word

  // Loop Closure Params
  unsigned p;                // Previous images to be discarded when searching for a loop
  float nndr_bf_lines;
  float nndr;                // Nearest neighbour distance ratio
  float nndr_bf;             // NNDR when matching with brute force
  double ep_dist;            // Distance to epipolar lines
  double conf_prob;          // Confidence probability
  double min_score;          // Min score to consider an image matching as correct
  unsigned island_size;      // Max number of images of an island
  unsigned min_inliers;      // Minimum number of inliers to consider a loop
  unsigned nframes_after_lc; // Number of frames after a lc to wait for new lc
  int min_consecutive_loops; // Min consecutive loops to avoid ep. geometry
  float non_sim_candidates;  // Weight to penalize score from islands which do not appear in Line and pts candidates 

  //Geometric Params
  bool b_l_endpts;
  bool b_l_inters_pts;
  bool b_l_global_rot;
  bool b_l_center_pts;

  // Others Params
  cv::Mat gt_matrix;        // Matrix which store the Image correspondences
  bool debug_loops;

  std::string output_path;

  LogosParameters logos_params;
  LPMParams lpm_params;
};

// LCDetectorStatus
enum LCDetectorStatus
{
  LC_DETECTED,
  LC_NOT_DETECTED,
  LC_NOT_ENOUGH_IMAGES,
  LC_NOT_ENOUGH_ISLANDS,
  LC_NOT_ENOUGH_INLIERS,
  LC_TRANSITION
};

// LCDetectorResult
struct LCDetectorResult
{
  LCDetectorResult() : status(LC_NOT_DETECTED),
                       query_id(1),
                       train_id(-1) {}

  inline bool isLoop()
  {
    return status == LC_DETECTED;
  }

  LCDetectorStatus status;
  unsigned query_id;
  unsigned train_id;
  unsigned inliers;
};

class LCDetector
{
public:
  explicit LCDetector(const LCDetectorParams &params);
  virtual ~LCDetector();

  void process(const unsigned image_id,
               const std::vector<cv::KeyPoint> &kps,
               const cv::Mat &descs,
               LCDetectorResult *result);
  void debug(const unsigned image_id,
             const std::vector<cv::KeyPoint> &kps,
             const cv::Mat &descs,
             std::ofstream &out_file);

  void debug(const unsigned image_id,
             const std::vector<cv::Mat> &v_images,
             const std::vector<cv::KeyPoint> &kps,
             const cv::Mat &descs,
             const std::vector<cv::line_descriptor::KeyLine> &keylines,
             const cv::Mat &descs_l,
             std::ofstream &out_file);

  bool DebugProposedIsland(const std::vector<cv::Mat> &v_images,
                              const int &query_idx,
                              const int &curr_idx,
                              const int &score,
                              int &display_time,
                              cv::Mat &concat);

  bool DebugProposedIslandWithMatches(
      const std::vector<cv::Mat> &v_images,
      const int &query_idx,
      const int &train_idx,
      const int &score_pts,
      const int &score_lines,
      const std::vector<std::vector<cv::KeyPoint>> &v_kps,
      const std::vector<std::vector<cv::line_descriptor::KeyLine>> &v_kls,
      const std::vector<DMatch> &v_matches,
      const std::vector<DMatch> &v_matches_l,
      int &display_time,
      cv::Mat &matched_img);

  bool checkOrient(const std::vector<cv::line_descriptor::KeyLine> &query_kls, 
const std::vector<cv::line_descriptor::KeyLine> &train_kls, const DMatch &match);

  void SearchVocCand(const cv::Mat & descs, std::vector<obindex2::ImageMatch> &image_matches_filt);

  void SearchVocCandLines(const cv::Mat & descs, std::vector<obindex2::ImageMatch> &image_matches_filt);

cv::Mat DrawLineNPtsMatches(
    std::vector<cv::Mat> v_imgs,
    const int &query_idx,
    const int &train_idx,
    const std::vector<std::vector<KeyLine>> &v_kls,
    const std::vector<std::vector<KeyPoint>> &v_kps,
    const std::vector<DMatch> &kls_matches,
    const std::vector<DMatch> &kpts_matches);

  cv::Mat draw2DLines(const cv::Mat gray_img,
                                const std::vector<KeyLine> &keylines);

  void setOutPMatSize(const int &size)
  {
    output_mat_ = cv::Mat::zeros(cv::Size(size, size), CV_8U);
  }

  cv::Mat getOutPMat()
  {
    return output_mat_;
  }

  bool ScoreCompQueryAdapt(const cv::Mat &v_pts_score,
                           const cv::Mat &v_lines_score,
                           std::vector<float> &v_scores);

  float computeIntegral(const cv::Mat &v_pts_score);


  std::vector<double> v_time_upd_voc_sthreat_pts_;
  std::vector<double> v_time_upd_voc_sthreat_l_;
  std::vector<double> v_time_upd_voc_multithreat_;

  std::vector<double> v_time_search_voc_sthreat_pts_;
  std::vector<double> v_time_search_voc_sthreat_l_;
  std::vector<double> v_time_search_voc_multithreat_;

  std::vector<double> v_time_merge_cand_island_select_;
  std::vector<double> v_time_spatial_ver_;

  std::vector<double> v_line_inliers_;


private:

  cv::Mat output_mat_;
  // Parameters
  unsigned p_;
  float nndr_bf_lines_;
  float nndr_;
  float nndr_bf_;
  float non_sim_candidates_;
  double ep_dist_;
  double conf_prob_;
  double min_score_;
  bool debug_loops_;
  unsigned island_size_;
  unsigned island_offset_;
  unsigned min_inliers_;
  unsigned nframes_after_lc_;
  GeomParams geom_params_;

  int num_incorrect_match;
  int num_not_found_match;

  // Last loop closure detected
  LCDetectorResult last_lc_result_;
  Island last_lc_island_;
  int min_consecutive_loops_;
  int consecutive_loops_;
  bool b_use_last_island_;

  // Ground Truth Matrix
  cv::Mat gt_matrix_;

  std::string wrong_matches_path_; 
  std::string not_found_matches_path_;

  // Image Index
  std::shared_ptr<obindex2::ImageIndex> index_;
  std::shared_ptr<obindex2::ImageIndex> index_l_;

  // Queues to delay the publication of hypothesis
  std::queue<unsigned> queue_ids_;
  std::queue<std::vector<cv::KeyPoint>> queue_kps_;
  std::queue<cv::Mat> queue_descs_;

  std::vector<std::vector<cv::KeyPoint>> prev_kps_;
  std::vector<cv::Mat> prev_descs_;

  std::vector<std::vector<cv::KeyPoint>> prev_kps_l_;
  std::vector<std::vector<cv::line_descriptor::KeyLine>> prev_kls_;
  std::vector<cv::Mat> prev_descs_l_;

  //LOGOS Parameters
  LogosParameters logos_params_;
  LPMParams lpm_params_;

  void addImage(const unsigned image_id,
                const std::vector<cv::KeyPoint> &kps,
                const cv::Mat &descs);

  void addImages(const unsigned image_id,
                const std::vector<cv::KeyPoint> &kps,
                const cv::Mat &descs);

  void addImageLines(const unsigned image_id,
                     const std::vector<cv::KeyPoint> &kps_l,
                     const cv::Mat &descs_l);
  void addImage(const unsigned image_id,
                const std::vector<cv::KeyPoint> &kps,
                const cv::Mat &descs,
                const std::vector<cv::KeyPoint> &kps_l,
                const cv::Mat &descs_l);

  void filterMatches(
      const std::vector<std::vector<cv::DMatch>> &matches_feats,
      std::vector<cv::DMatch> *matches);
  void filterCandidates(
      const std::vector<obindex2::ImageMatch> &image_matches,
      std::vector<obindex2::ImageMatch> *image_matches_filt);
  void buildIslands(
      const std::vector<obindex2::ImageMatch> &image_matches,
      std::vector<Island> *islands);
  void getPriorIslands(
      const Island &island,
      const std::vector<Island> &islands,
      std::vector<Island> *p_islands);
  unsigned checkEpipolarGeometry(
      const std::vector<cv::Point2f> &query,
      const std::vector<cv::Point2f> &train);
  void ratioMatchingBF(const cv::Mat &query,
                       const cv::Mat &train,
                       std::vector<cv::DMatch> *matches);

  void MatchingBF(const cv::Mat &query,
                  const cv::Mat &train,
                  std::vector<cv::DMatch> *matches);

  void ratioMatchingBFLines(
      const std::vector<cv::line_descriptor::KeyLine> &query_kls,
      const std::vector<cv::line_descriptor::KeyLine> &train_kls,
      const cv::Mat &query,
      const cv::Mat &train,
      std::vector<cv::DMatch> *matches);

  void ratioMatchingBFLNoGeom(const cv::Mat &query,
                                 const cv::Mat &train,
                                 std::vector<cv::DMatch> *matches);
  void convertPoints(const std::vector<cv::KeyPoint> &query_kps,
                     const std::vector<cv::KeyPoint> &train_kps,
                     const std::vector<cv::DMatch> &matches,
                     std::vector<cv::Point2f> *query,
                     std::vector<cv::Point2f> *train);
  void convertPoints(const std::vector<cv::KeyPoint> &query_kps,
                     const std::vector<cv::KeyPoint> &train_kps,
                     const std::vector<cv::DMatch> &matches,
                     const std::vector<cv::KeyPoint> &query_kps_l,
                     const std::vector<cv::KeyPoint> &train_kps_l,
                     const std::vector<cv::DMatch> &matches_l,
                     std::vector<cv::Point2f> *query,
                     std::vector<cv::Point2f> *train);

  double GlobalRotationImagePair(const std::vector<KeyLine> q_lines, const std::vector<KeyLine> &tr_lines);
  double getNormL2(double *arr, int size);
  void arrayMultiRatio(double *arr, int size, double ratio);

  void PtsNLineComb(const std::vector<KeyLine> &p_kls,
                    const std::vector<KeyLine> &c_kls,
                    const std::vector<cv::DMatch> &matches,
                    const int &kpts_size,
                    std::vector<cv::KeyPoint> &p_kpts,
                    std::vector<cv::KeyPoint> &c_kpts,
                    std::vector<cv::DMatch> &out_matches);

  void match2LineConv(const std::vector<cv::DMatch> &gms_matches,
                      const std::vector<cv::DMatch> &bf_matches,
                      const int &kpts_size, std::vector<cv::DMatch> &l_matches);

  cv::Mat DrawMatches(const std::vector<KeyLine> &linesInRight, const std::vector<KeyLine> &linesInLeft, const cv::Mat &r_image, const cv::Mat &l_image, const std::vector<DMatch> &matchResult);
};

} // namespace ibow_lcd

#endif // INCLUDE_IBOW_LCD_LCDETECTOR_H_
