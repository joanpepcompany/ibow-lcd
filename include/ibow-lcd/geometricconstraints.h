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

#ifndef INCLUDE_IBOW_GEOMETRIC_CONSTRAINTS_H_
#define INCLUDE_IBOW_GEOMETRIC_CONSTRAINTS_H_

#include <string>
#include <sstream>

#include "opencv2/core/core.hpp"
#include <opencv2/opencv.hpp>

#include <opencv2/line_descriptor.hpp>
using namespace cv::line_descriptor;
using namespace cv;

namespace ibow_lcd
{

struct GeomParams
{
    GeomParams() : b_l_endpts_(false),
                   b_l_inters_pts_(false),
                   b_l_global_rot_(false),
                   b_l_center_pt_(false),
                   ep_dist_(1.0),
                   conf_prob_(0.99)
    {}

    GeomParams(const GeomParams &t) : b_l_endpts_(t.b_l_endpts_),
                                      b_l_inters_pts_(t.b_l_inters_pts_),
                                      b_l_global_rot_(t.b_l_global_rot_),
                                      b_l_center_pt_(t.b_l_center_pt_),
                                      ep_dist_(t.ep_dist_),
                                      conf_prob_(t.conf_prob_)
    {}

    bool b_l_endpts_;
    bool b_l_inters_pts_;
    bool b_l_global_rot_;
    bool b_l_center_pt_;
    double ep_dist_;
    double conf_prob_;

};

// Intersect Point From Pair Lines
struct IntersectPt
{
    IntersectPt(const cv::Point2f &pt, const KeyLine &l1, const KeyLine &l2, const int &idx1, const int &idx2) : m_pt(pt), m_l1(l1), m_l2(l2), m_idx1(idx1), m_idx2(idx2) {}

    const cv::Point2f m_pt;
    const KeyLine m_l1;
    const KeyLine m_l2;
    const int m_idx1;
    const int m_idx2;
};
class GeomConstr
{
public:
    GeomConstr(const cv::Mat &q_img,
               const cv::Mat &t_img,
               const std::vector<cv::KeyPoint> &q_kps,
               const std::vector<cv::KeyPoint> &t_kps,
               const cv::Mat &q_descs_pts,
               const cv::Mat &t_descs_pts,
               const std::vector<cv::DMatch> &matches_pts,
               const std::vector<KeyLine> &q_kls,
               const std::vector<KeyLine> &t_kls,
               const cv::Mat &q_descs_l,
               const cv::Mat &t_descs_l,
               const std::vector<cv::DMatch> &matches_l,
               const GeomParams &geom_params);

     cv::Mat DrawMatches(const std::vector<KeyLine> &tr_lines,
                        const std::vector<KeyLine> &q_lines,
                        const cv::Mat &train_img,
                        const cv::Mat &query_img,
                        const std::vector<DMatch> &matchResult);

    inline int getLinesInliers()
    {
        return lines_inliers_;
    }

    inline int getPtsInliers()
    {
        return pts_inliers_;
    }

private:
    void FilterOrientMatches(const std::vector<KeyLine> q_lines,
                             const std::vector<KeyLine> &tr_lines,
                             const std::vector<cv::DMatch> &matches,
                             std::vector<cv::DMatch> &filt_matches,
                             std::vector<cv::DMatch> &non_filt_matches);

    double GlobalRotationImagePair(const std::vector<KeyLine> q_lines,
                                   const std::vector<KeyLine> &tr_lines);

    void arrayMultiRatio(double *arr, int size, double ratio);
    double getNormL2(double *arr, int size);

    void convertPoints(const std::vector<cv::KeyPoint> &query_kps,
                       const std::vector<cv::KeyPoint> &train_kps,
                       const std::vector<cv::DMatch> &matches,
                       std::vector<cv::Point2f> *query,
                       std::vector<cv::Point2f> *train);

    void detAndMatchIntersectPts(
        const std::vector<KeyLine> &q_kls,
        const std::vector<KeyLine> &t_kls,
        const std::vector<int> &v_matches,
        std::vector<IntersectPt> &v_p_inters_match_pt,
        std::vector<IntersectPt> &v_c_inters_match_pt);

    void getLineIntersectCand(
        const std::vector<KeyLine> &v_keylines,
        std::vector<std::pair<int, int>> &pair_cand,
        const float &distance_th,
        std::vector<cv::Point2f> &v_intersect_pts);

    float getEuclLineDistances(cv::Point l1_st, cv::Point l1_end,
                               cv::Point l2_st, cv::Point l2_end);

    void GetPtsWithTwoLineMatches(
        const std::vector<std::pair<int, int>> &v_c_pair_cand,
        const std::vector<std::pair<int, int>> &v_p_pair_cand, const std::vector<int> &matches,
        const std::vector<KeyLine> &c_keylines,
        const std::vector<KeyLine> &p_keylines,
        const std::vector<cv::Point2f> &v_c_intersect_pts,
        const std::vector<cv::Point2f> &v_p_intersect_pts,
        std::vector<IntersectPt> &v_c_inters_match_pt,
        std::vector<IntersectPt> &v_p_inters_match_pt);

    cv::Mat draw2DLines(const cv::Mat gray_img,
                        const std::vector<KeyLine> &keylines,
                        std::string img_name);

   

    cv::Mat DrawMatches(const std::vector<KeyLine> &tr_lines,
                        const std::vector<KeyLine> &q_lines,
                        const cv::Mat &train_img,
                        const cv::Mat &query_img,
                        const std::vector<int> &matchResult);

private:
    cv::Size img_size_;
    const cv::Mat &q_img_;
    const cv::Mat &t_img_;
    int pts_inliers_;
    int lines_inliers_;
    bool debug_results_;

    GeomParams geom_params_;
};

} // namespace ibow_lcd

#endif // INCLUDE_IBOW_GEOMETRIC_CONSTRAINTS_H_
