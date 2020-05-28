/*
 *  This file is part of the MR utility library.
 *
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This code is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this code; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/* Copyright (C) 2019-2020 Max-Planck-Society
   Author: Martin Reinecke */

#ifndef MRUTIL_MAV_H
#define MRUTIL_MAV_H

#include <cstdlib>
#include <array>
#include <vector>
#include <memory>
#include <numeric>
#include "mr_util/infra/error_handling.h"

namespace mr {

namespace detail_mav {

using namespace std;

template<typename T> class membuf
  {
  protected:
    using Tsp = shared_ptr<vector<T>>;
    Tsp ptr;
    const T *d;
    bool rw;

    membuf(const T *d_, membuf &other)
      : ptr(other.ptr), d(d_), rw(other.rw) {}
    membuf(const T *d_, const membuf &other)
      : ptr(other.ptr), d(d_), rw(false) {}

    // externally owned data pointer
    membuf(T *d_, bool rw_=false)
      : d(d_), rw(rw_) {}
    // externally owned data pointer, nonmodifiable
    membuf(const T *d_)
      : d(d_), rw(false) {}
    // allocate own memory
    membuf(size_t sz)
      : ptr(make_unique<vector<T>>(sz)), d(ptr->data()), rw(true) {}
    // share another memory buffer, but read-only
    membuf(const membuf &other)
      : ptr(other.ptr), d(other.d), rw(false) {}
#if defined(_MSC_VER)
    // MSVC is broken
    membuf(membuf &other)
      : ptr(other.ptr), d(other.d), rw(other.rw) {}
    membuf(membuf &&other)
      : ptr(move(other.ptr)), d(move(other.d)), rw(move(other.rw)) {}
#else
    // share another memory buffer, using the same read/write permissions
    membuf(membuf &other) = default;
    // take over another memory buffer
    membuf(membuf &&other) = default;
#endif

  public:
    // read/write access to element #i
    template<typename I> T &vraw(I i)
      {
      MR_assert(rw, "array is not writable");
      return const_cast<T *>(d)[i];
      }
    // read access to element #i
    template<typename I> const T &operator[](I i) const
      { return d[i]; }
    // read/write access to data area
    const T *data() const
      { return d; }
    // read access to data area
    T *vdata()
      {
      MR_assert(rw, "array is not writable");
      return const_cast<T *>(d);
      }
    bool writable() const { return rw; }
  };

class fmav_info
  {
  public:
    using shape_t = vector<size_t>;
    using stride_t = vector<ptrdiff_t>;

  protected:
    shape_t shp;
    stride_t str;
    size_t sz;

    static stride_t shape2stride(const shape_t &shp)
      {
      auto ndim = shp.size();
      stride_t res(ndim);
      res[ndim-1]=1;
      for (size_t i=2; i<=ndim; ++i)
        res[ndim-i] = res[ndim-i+1]*ptrdiff_t(shp[ndim-i+1]);
      return res;
      }
    template<typename... Ns> ptrdiff_t getIdx(size_t dim, size_t n, Ns... ns) const
      { return str[dim]*ptrdiff_t(n) + getIdx(dim+1, ns...); }
    ptrdiff_t getIdx(size_t dim, size_t n) const
      { return str[dim]*ptrdiff_t(n); }

  public:
    fmav_info(const shape_t &shape_, const stride_t &stride_)
      : shp(shape_), str(stride_), sz(accumulate(shp.begin(),shp.end(),size_t(1),multiplies<>()))
      {
      MR_assert(shp.size()>0, "at least 1D required");
      MR_assert(shp.size()==str.size(), "dimensions mismatch");
      }
    fmav_info(const shape_t &shape_)
      : fmav_info(shape_, shape2stride(shape_)) {}
    size_t ndim() const { return shp.size(); }
    size_t size() const { return sz; }
    const shape_t &shape() const { return shp; }
    size_t shape(size_t i) const { return shp[i]; }
    const stride_t &stride() const { return str; }
    const ptrdiff_t &stride(size_t i) const { return str[i]; }
    bool last_contiguous() const
      { return (str.back()==1); }
    bool contiguous() const
      {
      auto ndim = shp.size();
      ptrdiff_t stride=1;
      for (size_t i=0; i<ndim; ++i)
        {
        if (str[ndim-1-i]!=stride) return false;
        stride *= ptrdiff_t(shp[ndim-1-i]);
        }
      return true;
      }
    bool conformable(const fmav_info &other) const
      { return shp==other.shp; }
    template<typename... Ns> ptrdiff_t idx(Ns... ns) const
      {
      MR_assert(ndim()==sizeof...(ns), "incorrect number of indices");
      return getIdx(0, ns...);
      }
  };

template<size_t ndim> class mav_info
  {
  protected:
    static_assert(ndim>0, "at least 1D required");

    using shape_t = array<size_t, ndim>;
    using stride_t = array<ptrdiff_t, ndim>;

    shape_t shp;
    stride_t str;
    size_t sz;

    static stride_t shape2stride(const shape_t &shp)
      {
      stride_t res;
      res[ndim-1]=1;
      for (size_t i=2; i<=ndim; ++i)
        res[ndim-i] = res[ndim-i+1]*ptrdiff_t(shp[ndim-i+1]);
      return res;
      }
    template<typename... Ns> ptrdiff_t getIdx(size_t dim, size_t n, Ns... ns) const
      { return str[dim]*n + getIdx(dim+1, ns...); }
    ptrdiff_t getIdx(size_t dim, size_t n) const
      { return str[dim]*n; }

  public:
    mav_info(const shape_t &shape_, const stride_t &stride_)
      : shp(shape_), str(stride_), sz(accumulate(shp.begin(),shp.end(),size_t(1),multiplies<>())) {}
    mav_info(const shape_t &shape_)
      : mav_info(shape_, shape2stride(shape_)) {}
    size_t size() const { return sz; }
    const shape_t &shape() const { return shp; }
    size_t shape(size_t i) const { return shp[i]; }
    const stride_t &stride() const { return str; }
    const ptrdiff_t &stride(size_t i) const { return str[i]; }
    bool last_contiguous() const
      { return (str.back()==1); }
    bool contiguous() const
      {
      ptrdiff_t stride=1;
      for (size_t i=0; i<ndim; ++i)
        {
        if (str[ndim-1-i]!=stride) return false;
        stride *= ptrdiff_t(shp[ndim-1-i]);
        }
      return true;
      }
    bool conformable(const mav_info &other) const
      { return shp==other.shp; }
    bool conformable(const shape_t &other) const
      { return shp==other; }
    template<typename... Ns> ptrdiff_t idx(Ns... ns) const
      {
      static_assert(ndim==sizeof...(ns), "incorrect number of indices");
      return getIdx(0, ns...);
      }
  };


class FmavIter
  {
  private:
    fmav_info::shape_t pos;
    fmav_info arr;
    ptrdiff_t p;
    size_t rem;

  public:
    FmavIter(const fmav_info &arr_)
      : pos(arr_.ndim(), 0), arr(arr_), p(0), rem(arr_.size()) {}
    void advance()
      {
      --rem;
      for (int i_=int(pos.size())-1; i_>=0; --i_)
        {
        auto i = size_t(i_);
        p += arr.stride(i);
        if (++pos[i] < arr.shape(i))
          return;
        pos[i] = 0;
        p -= ptrdiff_t(arr.shape(i))*arr.stride(i);
        }
      }
    ptrdiff_t ofs() const { return p; }
    size_t remaining() const { return rem; }
  };


// "mav" stands for "multidimensional array view"
template<typename T> class fmav: public fmav_info, public membuf<T>
  {
  protected:
    using tbuf = membuf<T>;
    using tinfo = fmav_info;

    template<typename Func> void applyHelper(size_t idim, ptrdiff_t idx, Func func)
      {
      auto ndim = tinfo::ndim();
      if (idim+1<ndim)
        for (size_t i=0; i<shp[idim]; ++i)
          applyHelper<Func>(idim+1, idx+i*str[idim], func);
      else
        {
        T *d2 = vdata();
        for (size_t i=0; i<shp[idim]; ++i)
          func(d2[idx+i*str[idim]]);
        }
      }
    template<typename Func> void applyHelper(size_t idim, ptrdiff_t idx, Func func) const
      {
      auto ndim = tinfo::ndim();
      if (idim+1<ndim)
        for (size_t i=0; i<shp[idim]; ++i)
          applyHelper<Func>(idim+1, idx+i*str[idim], func);
      else
        {
        const T *d2 = data();
        for (size_t i=0; i<shp[idim]; ++i)
          func(d2[idx+i*str[idim]]);
        }
      }

    auto subdata(const shape_t &i0, const shape_t &extent) const
      {
      auto ndim = tinfo::ndim();
      shape_t nshp(ndim);
      stride_t nstr(ndim);
      ptrdiff_t nofs;
      MR_assert(i0.size()==ndim, "bad domensionality");
      MR_assert(extent.size()==ndim, "bad domensionality");
      size_t n0=0;
      for (auto x:extent) if (x==0) ++n0;
      nofs=0;
      nshp.resize(ndim-n0);
      nstr.resize(ndim-n0);
      for (size_t i=0, i2=0; i<ndim; ++i)
        {
        MR_assert(i0[i]<shp[i], "bad subset");
        nofs+=i0[i]*str[i];
        if (extent[i]!=0)
          {
          MR_assert(i0[i]+extent[i2]<=shp[i], "bad subset");
          nshp[i2] = extent[i]; nstr[i2]=str[i];
          ++i2;
          }
        }
      return make_tuple(nshp, nstr, nofs);
      }

  public:
    using tbuf::vraw, tbuf::operator[], tbuf::vdata, tbuf::data;

    fmav(const T *d_, const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), tbuf(d_) {}
    fmav(const T *d_, const shape_t &shp_)
      : tinfo(shp_), tbuf(d_) {}
    fmav(T *d_, const shape_t &shp_, const stride_t &str_, bool rw_)
      : tinfo(shp_, str_), tbuf(d_,rw_) {}
    fmav(T *d_, const shape_t &shp_, bool rw_)
      : tinfo(shp_), tbuf(d_,rw_) {}
    fmav(const shape_t &shp_)
      : tinfo(shp_), tbuf(size()) {}
    fmav(const T* d_, const tinfo &info)
      : tinfo(info), tbuf(d_) {}
    fmav(T* d_, const tinfo &info, bool rw_=false)
      : tinfo(info), tbuf(d_, rw_) {}
#if defined(_MSC_VER)
    // MSVC is broken
    fmav(const fmav &other) : tinfo(other), tbuf(other) {};
    fmav(fmav &other) : tinfo(other), tbuf(other) {}
    fmav(fmav &&other) : tinfo(other), tbuf(other) {}
#else
    fmav(const fmav &other) = default;
    fmav(fmav &other) = default;
    fmav(fmav &&other) = default;
#endif
    fmav(tbuf &buf, const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), tbuf(buf) {}
    fmav(const tbuf &buf, const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), tbuf(buf) {}
    fmav(const shape_t &shp_, const stride_t &str_, const T *d_, tbuf &buf)
      : tinfo(shp_, str_), tbuf(d_, buf) {}
    fmav(const shape_t &shp_, const stride_t &str_, const T *d_, const tbuf &buf)
      : tinfo(shp_, str_), tbuf(d_, buf) {}

    template<typename... Ns> const T &operator()(Ns... ns) const
      { return operator[](idx(ns...)); }
    template<typename... Ns> T &v(Ns... ns)
      { return vraw(idx(ns...)); }

    fmav subarray(const shape_t &i0, const shape_t &extent)
      {
      auto [nshp, nstr, nofs] = subdata(i0, extent);
      return fmav(nshp, nstr, tbuf::d+nofs, *this);
      }
    fmav subarray(const shape_t &i0, const shape_t &extent) const
      {
      auto [nshp, nstr, nofs] = subdata(i0, extent);
      return fmav(nshp, nstr, tbuf::d+nofs, *this);
      }
    template<typename Func> void apply(Func func)
      {
      if (contiguous())
        {
        T *d2 = vdata();
        for (auto v=d2; v!=d2+size(); ++v)
          func(*v);
        return;
        }
      applyHelper<Func>(0, 0,func);
      }
    template<typename Func> void apply(Func func) const
      {
      if (contiguous())
        {
        const T *d2 = data();
        for (auto v=d2; v!=d2+size(); ++v)
          func(*v);
        return;
        }
      applyHelper<Func>(0, 0, func);
      }
    vector<T> dump() const
      {
      vector<T> res(sz);
      size_t ii=0;
      apply([&](const T&v){res[ii++]=v;});
      return res;
      }
    void load (const vector<T> &v)
      {
      MR_assert(v.size()==sz, "bad input data size");
      size_t ii=0;
      apply([&](T &val){val=v[ii++];});
      }
  };

template<typename T, size_t ndim> class mav: public mav_info<ndim>, public membuf<T>
  {
//  static_assert((ndim>0) && (ndim<4), "only supports 1D, 2D, and 3D arrays");

  protected:
    using tinfo = mav_info<ndim>;
    using tbuf = membuf<T>;
    using typename tinfo::shape_t;
    using typename tinfo::stride_t;
    using tinfo::shp, tinfo::str;

    template<size_t idim, typename Func> void applyHelper(ptrdiff_t idx, Func func)
      {
      if constexpr (idim+1<ndim)
        for (size_t i=0; i<shp[idim]; ++i)
          applyHelper<idim+1, Func>(idx+i*str[idim], func);
      else
        {
        T *d2 = vdata();
        for (size_t i=0; i<shp[idim]; ++i)
          func(d2[idx+i*str[idim]]);
        }
      }
    template<size_t idim, typename Func> void applyHelper(ptrdiff_t idx, Func func) const
      {
      if constexpr (idim+1<ndim)
        for (size_t i=0; i<shp[idim]; ++i)
          applyHelper<idim+1, Func>(idx+i*str[idim], func);
      else
        {
        const T *d2 = data();
        for (size_t i=0; i<shp[idim]; ++i)
          func(d2[idx+i*str[idim]]);
        }
      }
    template<size_t idim, typename T2, typename Func>
      void applyHelper(ptrdiff_t idx, ptrdiff_t idx2,
                       const mav<T2,ndim> &other, Func func)
      {
      if constexpr (idim==0)
        MR_assert(conformable(other), "dimension mismatch");
      if constexpr (idim+1<ndim)
        for (size_t i=0; i<shp[idim]; ++i)
          applyHelper<idim+1, T2, Func>(idx+i*str[idim],
                                        idx2+i*other.str[idim], other, func);
      else
        {
        T *d2 = vdata();
        const T2 *d3 = other.data();
        for (size_t i=0; i<shp[idim]; ++i)
          func(d2[idx+i*str[idim]],d3[idx2+i*other.str[idim]]);
        }
      }

    template<size_t nd2> auto subdata(const shape_t &i0, const shape_t &extent) const
      {
      array<size_t, nd2> nshp;
      array<ptrdiff_t, nd2> nstr;
      ptrdiff_t nofs;
      size_t n0=0;
      for (auto x:extent) if (x==0)++n0;
      MR_assert(n0+nd2==ndim, "bad extent");
      nofs=0;
      for (size_t i=0, i2=0; i<ndim; ++i)
        {
        MR_assert(i0[i]<shp[i], "bad subset");
        nofs+=i0[i]*str[i];
        if (extent[i]!=0)
          {
          MR_assert(i0[i]+extent[i2]<=shp[i], "bad subset");
          nshp[i2] = extent[i]; nstr[i2]=str[i];
          ++i2;
          }
        }
      return make_tuple(nshp, nstr, nofs);
      }

  public:
    using tbuf::vraw, tbuf::operator[], tbuf::vdata, tbuf::data;
    using tinfo::contiguous, tinfo::size, tinfo::idx, tinfo::conformable;

    mav(const T *d_, const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), tbuf(d_) {}
    mav(T *d_, const shape_t &shp_, const stride_t &str_, bool rw_=false)
      : tinfo(shp_, str_), tbuf(d_, rw_) {}
    mav(const T *d_, const shape_t &shp_)
      : tinfo(shp_), tbuf(d_) {}
    mav(T *d_, const shape_t &shp_, bool rw_=false)
      : tinfo(shp_), tbuf(d_, rw_) {}
    mav(const array<size_t,ndim> &shp_)
      : tinfo(shp_), tbuf(size()) {}
#if defined(_MSC_VER)
    // MSVC is broken
    mav(const mav &other) : tinfo(other), tbuf(other) {}
    mav(mav &other): tinfo(other), tbuf(other) {}
    mav(mav &&other): tinfo(other), tbuf(other) {}
#else
    mav(const mav &other) = default;
    mav(mav &other) = default;
    mav(mav &&other) = default;
#endif
    mav(const shape_t &shp_, const stride_t &str_, const T *d_, membuf<T> &mb)
      : mav_info<ndim>(shp_, str_), membuf<T>(d_, mb) {}
    mav(const shape_t &shp_, const stride_t &str_, const T *d_, const membuf<T> &mb)
      : mav_info<ndim>(shp_, str_), membuf<T>(d_, mb) {}
    operator fmav<T>() const
      {
      return fmav<T>(*this, {shp.begin(), shp.end()}, {str.begin(), str.end()});
      }
    operator fmav<T>()
      {
      return fmav<T>(*this, {shp.begin(), shp.end()}, {str.begin(), str.end()});
      }
    template<typename... Ns> const T &operator()(Ns... ns) const
      { return operator[](idx(ns...)); }
    template<typename... Ns> T &v(Ns... ns)
      { return vraw(idx(ns...)); }
    template<typename Func> void apply(Func func)
      {
      if (contiguous())
        {
        T *d2 = vdata();
        for (auto v=d2; v!=d2+size(); ++v)
          func(*v);
        return;
        }
      applyHelper<0,Func>(0,func);
      }
    template<typename T2, typename Func> void apply
      (const mav<T2, ndim> &other,Func func)
      { applyHelper<0,T2,Func>(0,0,other,func); }
    void fill(const T &val)
      { apply([val](T &v){v=val;}); }
    template<size_t nd2> mav<T,nd2> subarray(const shape_t &i0, const shape_t &extent)
      {
      auto [nshp, nstr, nofs] = subdata<nd2> (i0, extent);
      return mav<T,nd2> (nshp, nstr, tbuf::d+nofs, *this);
      }
    template<size_t nd2> mav<T,nd2> subarray(const shape_t &i0, const shape_t &extent) const
      {
      auto [nshp, nstr, nofs] = subdata<nd2> (i0, extent);
      return mav<T,nd2> (nshp, nstr, tbuf::d+nofs, *this);
      }
  };

template<typename T, size_t ndim> class MavIter
  {
  protected:
    fmav<T> mav;
    array<size_t, ndim> shp;
    array<ptrdiff_t, ndim> str;
    fmav_info::shape_t pos;
    ptrdiff_t idx_;
    bool done_;

    template<typename... Ns> ptrdiff_t getIdx(size_t dim, size_t n, Ns... ns) const
      { return str[dim]*n + getIdx(dim+1, ns...); }
    ptrdiff_t getIdx(size_t dim, size_t n) const
      { return str[dim]*n; }

  public:
    MavIter(const fmav<T> &mav_)
      : mav(mav_), pos(mav.ndim()-ndim,0), idx_(0), done_(false)
      {
      for (size_t i=0; i<ndim; ++i)
        {
        shp[i] = mav.shape(mav.ndim()-ndim+i);
        str[i] = mav.stride(mav.ndim()-ndim+i);
        }
      }
    MavIter(fmav<T> &mav_)
      : mav(mav_), pos(mav.ndim()-ndim,0), idx_(0), done_(false)
      {
      for (size_t i=0; i<ndim; ++i)
        {
        shp[i] = mav.shape(mav.ndim()-ndim+i);
        str[i] = mav.stride(mav.ndim()-ndim+i);
        }
      }
    bool done() const
      { return done_; }
    void inc()
      {
      for (ptrdiff_t i=mav.ndim()-ndim-1; i>=0; --i)
        {
        idx_+=mav.stride(i);
        if (++pos[i]<mav.shape(i)) return;
        pos[i]=0;
        idx_-=mav.shape(i)*mav.stride(i);
        }
      done_=true;
      }
    size_t shape(size_t i) const { return shp[i]; }
    template<typename... Ns> ptrdiff_t idx(Ns... ns) const
      { return idx_ + getIdx(0, ns...); }
    template<typename... Ns> const T &operator()(Ns... ns) const
      { return mav[idx(ns...)]; }
    template<typename... Ns> T &v(Ns... ns)
      { return mav.vraw(idx(ns...)); }
  };

}

using detail_mav::fmav_info;
using detail_mav::fmav;
using detail_mav::mav;
using detail_mav::FmavIter;
using detail_mav::MavIter;

}

#endif
